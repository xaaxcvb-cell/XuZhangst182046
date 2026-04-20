import torch
import torch.nn as nn

###

from torchmetrics import Accuracy
from torch.cuda.amp import autocast

from networks import BaseModule
from utils import HScore, GaussianMixtureModel, calculate_entropy, calculate_cosine_similarity, calculate_kld, kl_dirichlet, mask
from augmentation import get_tta_transforms


class GmmBaAdaptationModule(BaseModule): #基于 GMM 的在线无源自适应模块
    def __init__(self, datamodule, feature_dim=256, lr=1e-2, red_feature_dim=64, p_reject=0.5, N_init=30,
                 augmentation=False, lam=1, temperature=0.1, ckpt_dir='', pseudo_label_quality=0.0):  ### add  pseudo_label_quality=1.0  augmentation=True   p_reject=0.5
        super(GmmBaAdaptationModule, self).__init__(datamodule, feature_dim, lr, ckpt_dir)  #old  p_reject=0.5    N_init=30   

        self.ckpt_dir = ckpt_dir
        self.pseudo_label_quality = pseudo_label_quality      ###
        self.i_counter = 0                                  ###

        torch.set_printoptions(precision=2)                ### 全局打印精度

        # ---------- Dataset information ----------  #~~~~~~~~~~~~~
        self.source_class_num = datamodule.source_private_class_num + datamodule.shared_class_num    ###OPDA: 3 + 6
        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num

        print(f"source_class_num: {self.source_class_num}")              ###
        print(f"known_classes_num: {self.known_classes_num}")            ###


        # Additional feature reduction model
        self.feature_reduction = nn.Sequential(nn.Linear(feature_dim, red_feature_dim)).to(self.device)           ###
        #old self.feature_reduction = nn.Sequential(nn.Linear(feature_dim, red_feature_dim)).to(self.device)

        # ---------- GMM ----------
        self.gmm = GaussianMixtureModel(self.source_class_num)

        # ---------- Unknown mask ----------
        self.mask = mask(0.5 - p_reject / 2, 0.5 + p_reject / 2, N_init)    #old  self.mask = mask(0.5 - p_reject / 2, 0.5 + p_reject / 2, N_init)

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
            x, y = train_batch   ###y shape  [B]
            
            print(f"y shape: {y.shape}, y: {y.tolist()}")            ###

            # Determine ground truth for the ODA or OPDA scenario
            y = torch.where(y >= self.source_class_num, self.source_class_num, y)   ###torch.where(condition, A, B)  ture:A false:B
            y = y.to(self.device)

            #y_hat, feat_ext = self.forward(x)  #y_hat为模型预测概率，feat_ext中间层提取出来的特征向量   #.forward from basemodel from network
            y_hat, logits, feat_ext = self.forward(x)  # l  logits   shape [B,class_num]
            softplus = nn.Softplus()
            evidence_Di = softplus(logits)        ### ek in paper            shape [B, K] 
            #print(f"--------evidence_Di--------: {evidence_Di}")                                    ###

            alpha_Di = evidence_Di + 1.0           ###shape [B, K] 
            s_Di = alpha_Di.sum(dim=1, keepdim=True)   # [B, 1]  # Dirichlet strength     uncertainty u = K/s  #########
            #print(f"===========s_Di=============: {s_Di}") 

            ###
            _, y_hat_Di = torch.max(alpha_Di, dim=1) 

            ###pres for dirichlet
            #alpha_Di_clone_detached   = alpha_Di.clone().detach()              ####
            #_, y_hat_Di = torch.max(alpha_Di_clone_detached, dim=1)           ####


            K_Di = self.source_class_num
            u_Di = K_Di / (s_Di)                                         ### uncertainty u = K/s
            #print(f"--------uncertainty u--------: {u_Di}")             ###

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

                #print(f"pseudo_labels: {pseudo_labels}")           ### test pseudo_labels

                likelihood = likelihood.to(self.device)

            # ---------- Generate a mask and monitor the result ----------
            known_mask, unknown_mask, rejection_mask = self.mask.calculate_mask(likelihood) #用伪标签取得mask

            #print("known_mask:", known_mask)                            ###
            #print("unknown_mask:", unknown_mask)                        ###
            #print("rejection_mask:", rejection_mask)                    ###



            #print("known_mask:", known_mask)                            ###
            #print("unknown_mask:", unknown_mask)                        ###
            #print("rejection_mask:", rejection_mask)                    ###

            known_mask = known_mask.to(self.device)
            unknown_mask = unknown_mask.to(self.device)
            rejection_mask = rejection_mask.to(self.device)
            


            ###
            ratio_Di_unknown = 0.3 
            ratio_Di_known = 0.8              ###
             
         

            non_rejection_mask = ~rejection_mask  # for all known and unkown samples####

            # Assign unknown pseudo-labels
            pseudo_labels[unknown_mask] = self.source_class_num          #重要，将位置类别的值设为K+1（其实是K，因为从0开始）
            #print(f"pseudo_labels[known_mask]: {pseudo_labels[known_mask]}")           ### test pseudo_labels
            
            entropy_values_test = calculate_entropy(likelihood)
            entropy_values_test = torch.tensor(entropy_values_test)
            
            
            ###按u大小打印伪标签和标签
            print("==========pseudo_labels, y_hat_Di y, u_Di, entropy_values_test ==========: ") 
            data_all = torch.cat([
                pseudo_labels.unsqueeze(1).float(),
                y_hat_Di.unsqueeze(1).float(),
                y.unsqueeze(1).float(),
                u_Di,
                entropy_values_test.unsqueeze(1).float()
            ], dim=1)

            _, indices = torch.sort(data_all[:, -2], descending=True)   # 按最后2列（u_Di）降序排序
            #_, indices = torch.sort(data_all[:, -1], descending=True)   # 按最后一列（u_Di）降序排序
            data_all_sorted = data_all[indices]
            print(data_all_sorted)


            
            ###按u大小打印源域外伪标签和标签
            pseudo_labels_unknow = pseudo_labels[unknown_mask]                                           ###
            y_unknow = y[unknown_mask]                                          ###
            y_hat_Di_unknow  = y_hat_Di[unknown_mask]                                          ###
            u_Di_unknow = u_Di[unknown_mask]               

            entropy_values_test_unknow = entropy_values_test[unknown_mask] 
            entropy_values_test_unknow = torch.tensor(entropy_values_test_unknow)
            
            print("==========pseudo_labels_unknow, y_hat_Di_unknow, y_unknow, u_Di_unknow, entropy_values_test_unknow ==========: ") 
            data_unknow = torch.cat([
                pseudo_labels_unknow.unsqueeze(1).float(),
                y_hat_Di_unknow.unsqueeze(1).float(),
                y_unknow.unsqueeze(1).float(),
                u_Di_unknow,
                entropy_values_test_unknow.unsqueeze(1).float()
            ], dim=1)


            _, indices = torch.sort(data_unknow[:, -2], descending=True)   # 按最后2列（u_Di）降序排序
            #_, indices = torch.sort(data_unknow[:, -1], descending=True)   # 按最后一列（u_Di）降序排序
            data_unknow_sorted = data_unknow[indices]
            print(data_unknow_sorted)
            #print(torch.cat([pseudo_labels_unknow.unsqueeze(1).float(), y_hat_Di_unknow.unsqueeze(1).float(), y_unknow.unsqueeze(1).float(),u_Di_unknow], dim=1))

            ### 只取u_Di大于ratio_Di_unknown的部分作为新unknown_mask
            u_flat = u_Di_unknow.squeeze(-1)  
            _, indices = torch.sort(u_flat, descending=True)
            k = max(1, int(len(u_flat) * ratio_Di_unknown))              ### 至少保留1个
            topk_indices_in_unknown = indices[:k]
            unknown_indices = torch.nonzero(unknown_mask, as_tuple=True)[0]
            selected_indices = unknown_indices[topk_indices_in_unknown]
            new_unknown_mask = torch.zeros_like(unknown_mask, dtype=torch.bool)
            new_unknown_mask[selected_indices] = True
            unknown_mask = new_unknown_mask



            #u_Di_unknow_m = u_Di_unknow.mean()
            #mask_high = (u_Di_unknow > u_Di_unknow_m).squeeze(-1) 
            #unknown_indices = torch.nonzero(unknown_mask, as_tuple=True)[0]
            #selected_indices = unknown_indices[mask_high]
            #new_unknown_mask = torch.zeros_like(unknown_mask, dtype=torch.bool)
            #new_unknown_mask[selected_indices] = True
            #unknown_mask = new_unknown_mask



            ###按u大小打印源域内伪标签和标签
            pseudo_labels_know = pseudo_labels[known_mask]                                          ###
            y_know = y[known_mask]                                          ###
            y_hat_Di_know  = y_hat_Di[known_mask]                                          ###
            u_Di_know = u_Di[known_mask]                                          ###
            
            entropy_values_test_know = entropy_values_test[known_mask] 
            entropy_values_test_know = torch.tensor(entropy_values_test_know)

            #print("pseudo_labels_know:", pseudo_labels_know.shape)
            #print("y_hat_Di_know:", y_hat_Di_know.shape)
            #print("y_know:", y_know.shape)
            #print("u_Di_know:", u_Di_know.shape)
            #print("entropy_values_test_know:", entropy_values_test_know.shape)

            print("==========pseudo_labels_know, y_hat_Di_know, y_know, u_Di_know,entropy_values_test_know ==========: ") 
            data_know = torch.cat([
                pseudo_labels_know.unsqueeze(1).float(),
                y_hat_Di_know.unsqueeze(1).float(),
                y_know.unsqueeze(1).float(),
                u_Di_know,
                entropy_values_test_know.unsqueeze(1).float()
            ], dim=1)


            _, indices = torch.sort(data_know[:, -2], descending=True)   # 按最后2列（u_Di）降序排序
            #_, indices = torch.sort(data_know[:, -1], descending=True)   # 按最后一列（u_Di）降序排序
            data_know_sorted = data_know[indices]
            print(data_know_sorted)

            ### 只取u_Di小于ratio_Di_known的部分作为新known_mask
            u_flat = u_Di_know.squeeze(-1)  
            _, indices = torch.sort(u_flat, descending=False)
            k = max(1, int(len(u_flat) * ratio_Di_known))              ### 至少保留1个
            topk_indices_in_known = indices[:k]
            known_indices = torch.nonzero(known_mask, as_tuple=True)[0]
            selected_indices = known_indices[topk_indices_in_known]
            new_known_mask = torch.zeros_like(known_mask, dtype=torch.bool)
            new_known_mask[selected_indices] = True
            known_mask = new_known_mask




            #u_Di_know_m = u_Di_know.mean()
            #mask_low = (u_Di_know < u_Di_know_m).squeeze(-1)   
            #known_indices = torch.nonzero(known_mask, as_tuple=True)[0]
            #selected_indices = known_indices[mask_low]
            #new_known_mask = torch.zeros_like(known_mask, dtype=torch.bool)
            #new_known_mask[selected_indices] = True
            #known_mask = new_known_mask



            ### 再打印一遍
            pseudo_labels_unknow = pseudo_labels[unknown_mask]                                           ###
            y_unknow = y[unknown_mask]                                          ###
            y_hat_Di_unknow  = y_hat_Di[unknown_mask]                                          ###
            u_Di_unknow = u_Di[unknown_mask]                                          ###

            entropy_values_test_unknow = entropy_values_test[unknown_mask] 
            entropy_values_test_unknow = torch.tensor(entropy_values_test_unknow)

            print("==========new pseudo_labels_unknow, y_hat_Di_unknow, y_unknow, u_Di_unknow ==========: ") 
            data_unknow = torch.cat([
                pseudo_labels_unknow.unsqueeze(1).float(),
                y_hat_Di_unknow.unsqueeze(1).float(),
                y_unknow.unsqueeze(1).float(),
                u_Di_unknow,
                entropy_values_test_unknow.unsqueeze(1).float()
            ], dim=1)

            _, indices = torch.sort(data_unknow[:, -2], descending=True)   # 按最后2列（u_Di）降序排序
            #_, indices = torch.sort(data_unknow[:, -1], descending=True)   # 按最后一列（u_Di）降序排序
            data_unknow_sorted = data_unknow[indices]
            print(data_unknow_sorted)



            pseudo_labels_know = pseudo_labels[known_mask]                                          ###
            y_know = y[known_mask]                                          ###
            y_hat_Di_know  = y_hat_Di[known_mask]                                          ###
            u_Di_know = u_Di[known_mask]                                          ###

            entropy_values_test_know = entropy_values_test[known_mask] 
            entropy_values_test_know = torch.tensor(entropy_values_test_know)

            print("==========new pseudo_labels_know, y_hat_Di_know, y_know, u_Di_know ==========: ") 
            data_know = torch.cat([
                pseudo_labels_know.unsqueeze(1).float(),
                y_hat_Di_know.unsqueeze(1).float(),
                y_know.unsqueeze(1).float(),
                u_Di_know,
                entropy_values_test_know.unsqueeze(1).float()
            ], dim=1)

            _, indices = torch.sort(data_know[:, -2], descending=True)   # 按最后2列（u_Di）降序排序
            #_, indices = torch.sort(data_know[:, -1], descending=True)   # 按最后一列（u_Di）降序排序
            data_know_sorted = data_know[indices]
            print(data_know_sorted)







            #print(torch.cat([pseudo_labels_know.unsqueeze(1).float(), y_hat_Di_know.unsqueeze(1).float(), y_know.unsqueeze(1).float(),u_Di_know], dim=1))

            ###若采用真实标签，用真实标签取得mask
            true_known_mask = y != self.source_class_num                  ###
            true_unknown_mask = y == self.source_class_num                ###
            true_rejection_mask = true_known_mask | true_unknown_mask               ###
            true_known_mask = true_known_mask.to(self.device)
            true_unknown_mask = true_unknown_mask.to(self.device)
            true_rejection_mask = true_rejection_mask.to(self.device)


            ###
            print(f"pseudo_labels: {pseudo_labels}")              ### 
            print(f"yyyyyyyyyyyyy: {y}")                                      ###
            same_ratio = (pseudo_labels == y).float().mean().item()                 ###       
            print("same_ratio:", same_ratio * 100, "%")                    ### 

            # mask
            real_unk = (y == self.source_class_num)                   ### 
            real_kn = (y != self.source_class_num)                  ### 

            # unknown
            num_real_unk = real_unk.sum().item()            ### 
            if num_real_unk > 0:
                unk_same_ratio = (((pseudo_labels == self.source_class_num) & real_unk).sum().item()) / num_real_unk
            else:
                unk_same_ratio = 0.0
            
            # known 
            num_real_kn = real_kn.sum().item()            ### 
            if num_real_kn > 0:
                kn_same_ratio = (((pseudo_labels != self.source_class_num) & real_kn).sum().item()) / num_real_kn
            else:
                kn_same_ratio = 0.0

            print("Unknown same_ratio P/y:", unk_same_ratio * 100, "%")                     ### 
            print("not Unknown same_ratio P/y:", kn_same_ratio * 100, "%")                  ### 




            print(f"===============y_hat_Di===============: {y_hat_Di}")          ####
            print(f"==============y_true_hat==============: {y}")                 ####

            y_hat_Di_known = y_hat_Di[true_known_mask]
            y_true_hat_known = y[true_known_mask]
            y_hat_Di_known_accuracy = (y_hat_Di_known == y_true_hat_known).sum().float() / true_known_mask.sum()
            print(f"==========y_hat_Di_accuracy==========: {y_hat_Di_known_accuracy}") 
            #print("==========uncertainty u,y_hat_Di,y_true_hat,pseudo_labels==========: ") 
            #print(torch.cat([u_Di, y_hat_Di.unsqueeze(1).float(), y.unsqueeze(1).float(),pseudo_labels.unsqueeze(1).float()], dim=1))
            #print(f"--------uncertainty u,y_hat_Di,y_true_hat--------: {u_Di},{y_hat_Di},{y}")




            if self.pseudo_label_quality == 1.0:                                 ###若采用真实标签，用真实标签和mask替代
                pseudo_labels = y.clone()                                        ###
                known_mask = true_known_mask                                     ###
                unknown_mask = true_unknown_mask                                 ###
                rejection_mask = true_rejection_mask                             ###
                
            print(f"pseudo_labels: {pseudo_labels}")              ### 
            print(f"yyyyyyyyyyyyy: {y}")                                      ###
                #y_known_mask = (y < self.source_class_num)                      ###
                #pseudo_labels[y_known_mask] = y[y_known_mask]                  ###

                #unknown_mask = ~y_known_mask                                   ###
                #pseudo_labels[unknown_mask] = self.source_class_num            ###



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
                known_mask_rep = known_mask.repeat(2)       ###复制成原来2倍
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

            ### ---------- Dirichlet-evidence loss ----------        #####
            ###1



            one_hot_know = torch.nn.functional.one_hot(pseudo_labels_know, num_classes=self.source_class_num)
            one_hot_know_Di = one_hot_know.detach().float()

            #print(f"one_hot_know_Di: {one_hot_know_Di}")

            alpha_Di = alpha_Di[known_mask].float()
            s_Di = s_Di[known_mask].float()

            per_sample_L_Di1 = torch.sum((one_hot_know_Di - alpha_Di / s_Di) ** 2 + alpha_Di * (s_Di - alpha_Di) / (s_Di ** 2 * (s_Di + 1)), dim=1 )

            L_Di1 = torch.sum(per_sample_L_Di1)
            #L_Di1 = ((one_hot_know_Di - alpha_Di/s_Di)**2 + alpha_Di* (s_Di - alpha_Di) / (s_Di**2 * (s_Di + 1))).sum()

            ###old
            #y_hat_Di_know = y_hat[known_mask,:]                       # y_hat_Di: [N_known, C]，每行是 softmax 概率
            #pred_labels_know = torch.argmax(y_hat_Di_know, dim=1)          # [N_known]，每个样本一个类别索引
            #s_Di_know = s_Di[known_mask, :] 
            #one_hot_know_Di = torch.nn.functional.one_hot(pred_labels_know, num_classes=y_hat_Di_know.size(1)).float()    
            #L_Di1 = ((one_hot_know_Di - y_hat_Di_know)**2 + y_hat_Di_know * (1 - y_hat_Di_know) / (s_Di_know + 1)).sum()
            #print(f"======one_hot_know_Di=====: {one_hot_know_Di}") 
            #print(f"======y_hat_Di_know=====: {y_hat_Di_know}") 
            #print(f"======s_Di_know=====: {one_hot_know_Di}") 

            #for test
            #A = torch.sum((one_hot_know_Di-alpha_Di / s_Di)**2, axis=1, keepdims=True) 
            #B = torch.sum(alpha_Di*(s_Di - alpha_Di)/(s_Di*s_Di*(s_Di+1)), axis=1, keepdims=True) 
            #AB = torch.sum(A + B) 
            #print(f"A+B: {AB.item():.6f} \t L_Di1: {L_Di1.item():.6f}")


            ###2

            alpha_Di_tilde = one_hot_know_Di + (1 - one_hot_know_Di)*alpha_Di
            per_sample_L_Di2 = kl_dirichlet(alpha_Di_tilde,self.source_class_num)
            
            L_Di2 = torch.sum(per_sample_L_Di2)
            self.annealing_step = 200
            annealing_coef = min(1.0, self.global_step / self.annealing_step)
            
            a_L_Di2 = annealing_coef * L_Di2
            print(f"self.global_step:{self.global_step}") 
            print(f"per_sample_L_Di2: {per_sample_L_Di2}") 
            print(f"L_Di2: {L_Di2}") 
            print(f"annealing_coef: {annealing_coef}") 
            print(f"a_L_Di2: {a_L_Di2}")

            ###old
            #y_hat_Di_both = y_hat[rejection_mask,:]
            #pred_labels_both = torch.argmax(y_hat_Di_both, dim=1)
            #one_hot_both = torch.nn.functional.one_hot(pred_labels_both, num_classes=y_hat_Di_both.size(1)).float() 
            #alpha_Di_both = alpha_Di[rejection_mask,:]
            #alpha_Di_tilde = one_hot_both + (1 - one_hot_both)*alpha_Di_both
            #L_Di2 = kl_dirichlet(alpha_Di_tilde)                                        ###
            #print(f"-----------alpha_tilde----: {alpha_Di_tilde}")                          ###
            #print(f"-----------L_Di2----------: {L_Di2}")  

            #self.i_counter = self.i_counter + 1
            #lambda_t = min(1.0, self.i_counter / 10)
            #print(f"-----------self.trainer.current_epoch----------: {self.trainer.current_epoch}") 
            #print(f"-----------lambda_t----------: {lambda_t}")  
            #L_Di2 = lambda_t * L_Di2.sum()
            #print(f"============L_Di2============: {L_Di2}") 


            #self.loss = L_con + self.lam * L_kl   ###old    loss function
            #self.loss = - L_con + self.lam * L_kl  + 5*L_Di1 + 5*a_L_Di2
            #self.loss = - L_con + self.lam * L_kl  + 2*L_Di1 + 2*a_L_Di2
            self.loss = - L_con + self.lam * L_kl  + L_Di1 + a_L_Di2
            #self.loss = - L_con + self.lam * L_kl  + 0.1*L_Di1 + 0.1*a_L_Di2  ###loss function
            #self.loss = L_Di1 + a_L_Di2
            #self.loss = 0.5*L_Di1 + 0.5*a_L_Di2
            #self.loss = 0.1*L_Di1 + 0.1*a_L_Di2
            #self.loss = 0.05*L_Di1 + 0.05*a_L_Di2
            
            
            

            print(f"self.loss: {self.loss.item()}")                          ###
            print(f"L_con: {L_con.item()}")                                           ###
            print(f"L_kl: {L_kl.item()}")
            print(f"L_Di1: {L_Di1.item()}, L_Di2: {L_Di2.item()}")                       ###, L_Di2: {L_Di2.item()}
            #a = self.lam * L_kl
            #print(f"L_con: {L_con.item()}, self.lam * L_kl: {a.item()}")

        # log into progress bar
        self.log('train_loss', self.loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.total_train_acc, on_epoch=True, prog_bar=True)
        self.log('tta_acc', self.total_online_tta_acc, on_epoch=True, prog_bar=True)

        print(type(self.total_train_acc), self.total_train_acc)
        print(type(self.total_online_tta_acc), self.total_online_tta_acc)

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
