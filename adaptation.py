import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torch.cuda.amp import autocast

from networks import BaseModule
from utils import HScore, GaussianMixtureModel, calculate_entropy, calculate_cosine_similarity, calculate_kld, kl_dirichlet, mask
from augmentation import get_tta_transforms


class GmmBaAdaptationModule(BaseModule): #基于 GMM 的在线无源自适应模块
    def __init__(self, datamodule, feature_dim=256, lr=1e-2, red_feature_dim=64, p_reject=0.5, N_init=30,
                 augmentation=True, lam=1, temperature=0.1, ckpt_dir=''):
        super(GmmBaAdaptationModule, self).__init__(datamodule, feature_dim, lr, ckpt_dir)

        self.ckpt_dir = ckpt_dir

        # ---------- Dataset information ----------  #~~~~~~~~~~~~~
        self.source_class_num = datamodule.source_private_class_num + datamodule.shared_class_num
        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num

        # Additional feature reduction model
        self.feature_reduction = nn.Sequential(nn.Linear(feature_dim, red_feature_dim)).to(self.device)

        # ---------- GMM ----------
        self.gmm = GaussianMixtureModel(self.source_class_num)

        # ---------- Unknown mask ----------
        self.mask = mask(0.5 - p_reject / 2, 0.5 + p_reject / 2, N_init)

        # ---------- Further initializations ----------
        self.tta_transform = get_tta_transforms()
        self.augmentation = augmentation
        self.temperature = temperature
        self.lam = lam  #lam（lambda）权重系数 / 超参数

        # definition of h-score and accuracy
        self.total_train_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num)
        self.total_online_tta_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.total_online_tta_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)

    def configure_optimizers(self): #优化器配置
        # define different learning rates for different subnetworks
        params_group = []

        for k, v in self.backbone.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 0.1}]
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.feature_reduction.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]

        optimizer = torch.optim.SGD(params_group, momentum=0.9, nesterov=True)
        return optimizer

    def training_step(self, train_batch):   ###核心算法流程
        # ----------- Open-World Test-time Training ------------
        self.backbone.train()
        self.feature_extractor.train()
        self.classifier.train()

        with autocast(): #自动选择合适的数据精度（float16 / bfloat16 / float32）来执行运算
            x, y = train_batch

            # Determine ground truth for the ODA or OPDA scenario
            y = torch.where(y >= self.source_class_num, self.source_class_num, y)
            y = y.to(self.device)

            #y_hat, feat_ext = self.forward(x)  #y_hat为预测概率，feat_ext中间层提取出来的特征向量
            y_hat, logits, feat_ext = self.forward(x)  # l  logits   shape [B,class_num]
            ###alpha = torch.exp(logits) + 1           #Dirichlet parameter alpha
            alpha = torch.exp(torch.clamp(logits, min=-10, max=10)) + 1.0
            s = alpha.sum(dim=1, keepdim=True)   # [B, 1]  # Dirichlet strength     uncertainty u = K/s
            #y_hat_aug, feat_ext_aug = self.forward(self.tta_transform(x))
            y_hat_aug, logits_aug, feat_ext_aug = self.forward(self.tta_transform(x))

            with torch.no_grad():
                feat_ext = self.feature_reduction(feat_ext)
                feat_ext_aug = self.feature_reduction(feat_ext_aug)
                # Update the GMM
                y_hat_clone_detached = y_hat.clone().detach()
                self.gmm.soft_update(feat_ext, y_hat_clone_detached)

                _, pseudo_labels, likelihood = self.gmm.get_labels(feat_ext)
                pseudo_labels = pseudo_labels.to(self.device)
                likelihood = likelihood.to(self.device)

            # ---------- Generate a mask and monitor the result ----------
            known_mask, unknown_mask, rejection_mask = self.mask.calculate_mask(likelihood)

            known_mask = known_mask.to(self.device)
            unknown_mask = unknown_mask.to(self.device)
            rejection_mask = rejection_mask.to(self.device)

            non_rejection_mask = ~rejection_mask  # for all known and unkown samples

            # Assign unknown pseudo-labels
            pseudo_labels[unknown_mask] = self.source_class_num

            # ---------- Enable OPDA for predictions ----------
            _, preds = torch.max(y_hat_clone_detached, dim=1)
            unknown_threshold = (self.mask.tau_low + self.mask.tau_high) / 2

            entropy_values = calculate_entropy(likelihood)
            output_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
            output_mask[entropy_values >= unknown_threshold] = True
            preds[output_mask] = self.source_class_num

            # ---------- Update the H-Score ----------
            self.total_online_tta_acc(preds, y)
            self.total_online_tta_hscore.update(preds, y)

            # ---------- Calculate accuracy and loss ----------
            pseudo_labels_rejected = pseudo_labels[rejection_mask]

            if not self.open_flag:
                self.total_train_acc(y_hat[rejection_mask], pseudo_labels_rejected)

            # ---------- Calculate the loss -----------
            # ---------- Contrastive loss -----------
            feat_ext = feat_ext.to(self.device)
            logits = logits.to(self.device)                 # logits
            feat_ext_aug = feat_ext_aug.to(self.device)
            logits_aug = logits_aug.to(self.device)         # logits_aug
            if self.augmentation:
                feat_total = torch.cat([feat_ext, feat_ext_aug], dim=0)
            else:
                feat_total = feat_ext
            mu = self.gmm.mu.to(self.device)
            # Calculate all cosine similarities between features (embeddings)
            cos_feat_feat = torch.exp(calculate_cosine_similarity(feat_total, feat_total) / self.temperature)
            # Calculate all cosine similarities between features (embeddings) and GMM means
            cos_feat_mu = torch.exp(calculate_cosine_similarity(mu, feat_total) / self.temperature)

            # Minimize distance between known features and their corresponding mean
            # Maximize distance between known/unknown features and the mean of different classes
            divisor = torch.sum(cos_feat_mu, dim=0)
            logarithmus = torch.log(torch.divide(cos_feat_mu, divisor.unsqueeze(0)))
            if self.augmentation:
                known_mask_rep = known_mask.repeat(2)
                pseudo_labels_rep = pseudo_labels.repeat(2)
            else:
                known_mask_rep = known_mask
                pseudo_labels_rep = pseudo_labels
            used = torch.gather(logarithmus[known_mask_rep], 1, pseudo_labels_rep[known_mask_rep].view(-1, 1))
            L_mu_feat = torch.sum(torch.sum(used, dim=0))

            # Minimize distance between known features of the same class
            # Maximize distance between known/unknown features of different classes
            divisor = torch.sum(cos_feat_feat, dim=0)
            logarithmus = torch.log(torch.divide(cos_feat_feat, divisor.unsqueeze(0)))
            # Calculate the equality between elements of pseudo_label along both axes
            pseudo_label_rep_expanded = pseudo_labels_rep.unsqueeze(1)
            mask = pseudo_label_rep_expanded == pseudo_labels_rep
            used = torch.zeros_like(logarithmus)
            used[mask] = logarithmus[mask.bool()]
            L_feat_feat = torch.sum(torch.sum(used[known_mask_rep, known_mask_rep], dim=0))
            L_con = L_mu_feat + L_feat_feat

            # ---------- KL-Divergence loss ----------
            # Maximize divergence between uniform distribution and models output of known classes
            likelihood = y_hat[known_mask,:]
            true_dist = torch.ones_like(likelihood) / 1e3
            kl_known = - torch.sum(calculate_kld(likelihood, true_dist))

            # Minimize divergence between uniform distribution and models output of unknown classes
            likelihood = y_hat[unknown_mask,:]
            true_dist = torch.ones_like(likelihood) / 1e3
            kl_unknown = torch.sum(calculate_kld(likelihood, true_dist))

            L_kl = kl_known + kl_unknown

            # ---------- Dirichlet-evidence loss ----------
            #1
            y_hat_Di_know = y_hat[known_mask,:]                       # y_hat_Di: [N_known, C]，每行是 softmax 概率
            pred_labels_know = torch.argmax(y_hat_Di_know, dim=1)          # [N_known]，每个样本一个类别索引
            s_know = s[known_mask, :] 
            one_hot_know = torch.nn.functional.one_hot(pred_labels_know, num_classes=y_hat_Di_know.size(1)).float()    
            L_Di1 = ((one_hot_know - y_hat_Di_know)**2 + y_hat_Di_know * (1 - y_hat_Di_know) / (s_know + 1)).sum()
            #2
            y_hat_Di_both = y_hat[non_rejection_mask,:]
            pred_labels_both = torch.argmax(y_hat_Di_both, dim=1)
            one_hot_both = torch.nn.functional.one_hot(pred_labels_both, num_classes=y_hat_Di_both.size(1)).float() 
            alpha_both = alpha[non_rejection_mask,:]
            alpha_tilde = one_hot_both + (1 - one_hot_both)*alpha_both
            L_Di2 = kl_dirichlet(alpha_tilde)
            L_Di2 = L_Di2.mean()


            #self.loss = L_con + self.lam * L_kl   ###loss function
            self.loss = L_con + self.lam * L_kl + L_Di1 + L_Di2  ###loss function

        # log into progress bar
        self.log('train_loss', self.loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.total_train_acc, on_epoch=True, prog_bar=True)
        self.log('tta_acc', self.total_online_tta_acc, on_epoch=True, prog_bar=True)

        return self.loss

    def on_train_epoch_end(self):
        # ---------- Monitor the performance of the OPDA setting ----------
        print(self.total_online_tta_acc.compute())
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.total_online_tta_hscore.compute()
            print(f"H-Score (Epoch): {h_score}")
            print(f"Known Accuracy (Epoch): {known_acc}")
            print(f"Unknown Accuracy: {unknown_acc}")
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('Epoch UnknownAcc', unknown_acc)
