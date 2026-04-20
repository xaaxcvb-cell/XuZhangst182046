import torch
import torch.nn as nn

import wandb

wandb.init(project="Dirichlet", name="adaption")

###

from torchmetrics import Accuracy
from torch.cuda.amp import autocast

from networks import BaseModule
from utils import HScore, GaussianMixtureModel, calculate_entropy, calculate_cosine_similarity, calculate_kld, kl_dirichlet, DE_dirichlet, print_sorted,log_metrics, mask
from augmentation import get_tta_transforms


class GmmBaAdaptationModule(BaseModule): #基于 GMM 的在线无源自适应模块
    def __init__(self, datamodule, feature_dim=256, lr=1e-2, red_feature_dim=64, p_reject=0.5, N_init=30,
                 augmentation=False, lam=1, temperature=0.1, ckpt_dir='', pseudo_label_quality=1.0, Dirichlet=0.0):  ### add  pseudo_label_quality=1.0  augmentation=True   p_reject=0.5
        super(GmmBaAdaptationModule, self).__init__(datamodule, feature_dim, lr, ckpt_dir)  #old    N_init=30   

        self.ckpt_dir = ckpt_dir
        self.pseudo_label_quality = pseudo_label_quality      ###
        self.i_counter = 0                                  ###
        self.Dirichlet = Dirichlet

        self.num_correct_unknow_accum = 0           ###  用于精度统计的全局计数
        self.num_correct_know_accum = 0             ###
        self.num_all_unknow_accum = 0               ###
        self.num_all_know_accum = 0                 ###
        self.final_num_correct_unknow_accum = 0           ###  用于精度统计的全局计数
        self.final_num_correct_know_accum = 0             ###
        self.final_num_all_unknow_accum = 0               ###
        self.final_num_all_know_accum = 0                 ###

        torch.set_printoptions(precision=2)                ### 全局打印精度

        # ---------- Dataset information ----------  #~~~~~~~~~~~~~
        self.source_class_num = datamodule.source_private_class_num + datamodule.shared_class_num    ###OPDA: 3 + 6
        self.known_classes_num = datamodule.shared_class_num + datamodule.source_private_class_num

        print(f"source_class_num: {self.source_class_num}")              ###
        print(f"known_classes_num: {self.known_classes_num}")            ###


        # Additional feature reduction model
        self.feature_reduction = nn.Sequential(nn.Linear(feature_dim, red_feature_dim)).to(self.device)

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

    def configure_optimizers(self): #优化器配置,包含lr
        # define different learning rates for different subnetworks
        params_group = []

        for k, v in self.backbone.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 0.1}]
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier_di1.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 2}]
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

            ###
            ratio_Di1_unknown = 0.1            ###0.3     1 for original
            ratio_Di1_known = 0.8             ###0.8      1 for original
            ratio_Di2_unknown = 1            ###0.3     1 for original
            ratio_Di2_known = 1           ###0.8      1 for original


            x, y = train_batch   ###y shape  [B]

            # Determine ground truth for the ODA or OPDA scenario
            y = torch.where(y >= self.source_class_num, self.source_class_num, y)   ###torch.where(condition, A, B)  ture:A false:B
            y = y.to(self.device)

            ###提取真实标签
            true_known_mask = y != self.source_class_num                  ###
            true_unknown_mask = y == self.source_class_num                ###
            true_rejection_mask = true_known_mask | true_unknown_mask     ###
            true_known_mask = true_known_mask.to(self.device)
            true_unknown_mask = true_unknown_mask.to(self.device)
            true_rejection_mask = true_rejection_mask.to(self.device)

            #y_hat, feat_ext = self.forward(x)  #y_hat为模型预测概率，feat_ext中间层提取出来的特征向量   #.forward from basemodel from network
            y_hat, logits1, logits2, feat_ext = self.forward(x)  # l  logits   shape [B,class_num]

            if self.Dirichlet == 1.0:
                softplus = nn.Softplus()
                evidence_Di1 = softplus(logits1)        ### ek in paper            shape [B, K] 
                #print(f"--------evidence_Di1--------: {evidence_Di1}")                                    ###

                alpha_Di1 = evidence_Di1 + 1.0           ###shape [B, K] 
                s_Di1 = alpha_Di1.sum(dim=1, keepdim=True)   # [B, 1]  # Dirichlet strength     uncertainty u = K/s  #########
                #print(f"===========s_Di=============: {s_Di}") 

                alpha_Di2 = torch.exp(logits2)
                de_Di2 = DE_dirichlet(alpha_Di2)            ###

                ###
                _, y_hat_Di1 = torch.max(alpha_Di1, dim=1) 
                _, y_hat_Di2 = torch.max(alpha_Di2, dim=1)

                ###pres for dirichlet
                #alpha_Di1_clone_detached   = alpha_Di1.clone().detach()              ####
                #_, y_hat_Di1 = torch.max(alpha_Di1_clone_detached, dim=1)           ####


                K_Di1 = self.source_class_num
                u_Di1 = K_Di1 / (s_Di1)                                         ###(s_Di1)   uncertainty u = K/s
                u_Di1 = u_Di1.squeeze(1)                                      ###.squeeze(1)  和其他数据形状保持一致
                #print(f"--------uncertainty u--------: {u_Di1}")             ###

            #y_hat_aug, feat_ext_aug = self.forward(self.tta_transform(x))
            y_hat_aug, logits1_aug, logits2_aug, feat_ext_aug = self.forward(self.tta_transform(x))

            with torch.no_grad():
                feat_ext = self.feature_reduction(feat_ext)
                feat_ext_aug = self.feature_reduction(feat_ext_aug)
                # Update the GMM
                y_hat_clone_detached = y_hat.clone().detach()
                self.gmm.soft_update(feat_ext, y_hat_clone_detached)
                _, preds_test = torch.max(y_hat_clone_detached, dim=1)

                _, pseudo_labels, likelihood = self.gmm.get_labels(feat_ext)
                if self.pseudo_label_quality == 1.0:                                 ###若采用真实标签，用真实标签和mask替代
                    pseudo_labels = y.clone() 
                pseudo_labels = pseudo_labels.to(self.device)

                #print(f"pseudo_labels: {pseudo_labels}")           ### test pseudo_labels

                likelihood = likelihood.to(self.device)

            # ---------- Generate a mask and monitor the result ----------
            known_mask, unknown_mask, rejection_mask = self.mask.calculate_mask(likelihood) #用伪标签取得mask

            known_mask = known_mask.to(self.device)
            unknown_mask = unknown_mask.to(self.device)
            rejection_mask = rejection_mask.to(self.device)
            

            # Assign unknown pseudo-labels
            pseudo_labels[unknown_mask] = self.source_class_num          #重要，将位置类别的值设为K+1（其实是K，因为从0开始）
            #print(f"pseudo_labels[known_mask]: {pseudo_labels[known_mask]}")           ### test pseudo_labels
            
            #entropy_values_test = calculate_entropy(likelihood)
            #entropy_values_test = torch.tensor(entropy_values_test)

            pseudo_labels_unknow = pseudo_labels[unknown_mask]              ###
            y_unknow = y[unknown_mask]                                      ###
            pseudo_labels_know = pseudo_labels[known_mask]                  ###
            y_know = y[known_mask]                                          ###
            

            if self.Dirichlet == 1.0:
                ###按u大小打印batch内所有伪标签，Di预测标签，真实标签，de不确定度，u不确定度
                print_sorted(pseudo_labels, preds_test, y, de_Di2, u_Di1, descending=True, name="pseudo_labels, preds_test, y, de_Di2, u_Di1")

                ###按u大小打印batch内unknown伪标签，Di预测标签，真实标签，de不确定度，u不确定度
                                                        ###
                y_hat_Di1_unknow  = y_hat_Di1[unknown_mask]                                          ###
                de_Di2_unknow = de_Di2[unknown_mask]    
                u_Di1_unknow = u_Di1[unknown_mask]               

                #entropy_values_test_unknow = entropy_values_test[unknown_mask] 
                #entropy_values_test_unknow = torch.tensor(entropy_values_test_unknow)
                print_sorted(pseudo_labels_unknow, y_hat_Di1_unknow, y_unknow, de_Di2_unknow, u_Di1_unknow, descending=True, name="pseudo_labels_unknow, y_hat_Di1_unknow, y_unknow, de_Di2_unknow, u_Di1_unknow")
                #print(torch.cat([pseudo_labels_unknow.unsqueeze(1).float(), y_hat_Di_unknow.unsqueeze(1).float(), y_unknow.unsqueeze(1).float(),u_Di_unknow], dim=1))

                ###按u大小打印源域内伪标签和标签
                
                y_hat_Di1_know  = y_hat_Di1[known_mask]                                          ###
                de_Di2_know = de_Di2[known_mask] 
                u_Di1_know = u_Di1[known_mask]                                          ###

                print_sorted(pseudo_labels_know, y_hat_Di1_know, y_know, de_Di2_know, u_Di1_know, descending=True, name="pseudo_labels_know, y_hat_Di1_know, y_know, de_Di2_know, u_Di1_know")
            
                #entropy_values_test_know = entropy_values_test[known_mask] 
                #entropy_values_test_know = torch.tensor(entropy_values_test_know)


            ###打印原伪标签精度
            correct_unknow = (pseudo_labels_unknow == y_unknow).sum().item()
            correct_know = (pseudo_labels_know == y_know).sum().item()

            total_unknow = y_unknow.numel()
            total_know = y_know.numel()

            acc_pseudo_labels_unknow = correct_unknow / total_unknow if total_unknow > 0 else 0.0
            acc_pseudo_labels_know = correct_know / total_know if total_know > 0 else 0.0

            self.num_correct_unknow_accum += correct_unknow
            self.num_correct_know_accum += correct_know
            self.num_all_unknow_accum += total_unknow
            self.num_all_know_accum += total_know

            acc_pseudo_labels_unknow_accum = self.num_correct_unknow_accum / self.num_all_unknow_accum if self.num_all_unknow_accum > 0 else 0.0
            acc_pseudo_labels_know_accum = self.num_correct_know_accum / self.num_all_know_accum if self.num_all_know_accum > 0 else 0.0

            print(f"acc_pseudo_labels_unknow_step: {acc_pseudo_labels_unknow}")
            print(f"acc_pseudo_labels_know_step: {acc_pseudo_labels_know}")
            print(f"acc_pseudo_labels_unknow_accum: {acc_pseudo_labels_unknow_accum}")
            print(f"acc_pseudo_labels_know_accum: {acc_pseudo_labels_know_accum}")


            if self.Dirichlet == 1.0:
                ### 按不确定度更新unknown mask, Di1和Di2顺序可换
                ###1.按U（Di1）排列unknown，只取部分作为新unknown_mask
                u_flat = u_Di1_unknow.squeeze(-1)  
                _, indices = torch.sort(u_flat, descending=True)
                k = max(1, int(len(u_flat) * ratio_Di1_unknown))              ### 至少保留1个
                topk_indices_in_unknown = indices[:k]
                unknown_indices = torch.nonzero(unknown_mask, as_tuple=True)[0]
                selected_indices = unknown_indices[topk_indices_in_unknown]
                new_unknown_mask = torch.zeros_like(unknown_mask, dtype=torch.bool)   ###创建一个和 unknown_mask 形状完全一样的张量，但里面的值全部是 0
                new_unknown_mask[selected_indices] = True
                unknown_mask = new_unknown_mask

                ###2.按DE（Di2）排列unknown，只取部分作为新unknown_mask
                de_Di2_unknown = de_Di2[unknown_mask]   ###交换Di1和Di2顺序时要改

                u_flat = de_Di2_unknown 
                _, indices = torch.sort(u_flat, descending=True)
                k = max(1, int(len(u_flat) * ratio_Di2_unknown))              ### 至少保留1个
                topk_indices_in_unknown_2 = indices[:k]
                unknown_indices_2 = torch.nonzero(unknown_mask, as_tuple=True)[0]
                selected_indices_2 = unknown_indices_2[topk_indices_in_unknown_2]
                new_unknown_mask_2 = torch.zeros_like(unknown_mask, dtype=torch.bool)
                new_unknown_mask_2[selected_indices_2] = True
                unknown_mask = new_unknown_mask_2                     ###使用new_known_mask替代known_mask


                ### 按不确定度更新known mask, Di1和Di2顺序可换
                ###1.按U（Di1）排列known，只取部分作为新known_mask
                u_flat = u_Di1_know.squeeze(-1)  
                _, indices = torch.sort(u_flat, descending=False)
                k = max(1, int(len(u_flat) * ratio_Di1_known))              ### 至少保留1个
                topk_indices_in_known = indices[:k]
                known_indices = torch.nonzero(known_mask, as_tuple=True)[0]
                selected_indices = known_indices[topk_indices_in_known]
                new_known_mask = torch.zeros_like(known_mask, dtype=torch.bool)
                new_known_mask[selected_indices] = True
                known_mask = new_known_mask                     ###使用new_known_mask替代known_mask 

                ###2.按DE（Di2）排列unknown，只取部分作为新unknown_mask
                de_Di2_known = de_Di2[known_mask]         ###交换Di1和Di2顺序时要改

                u_flat = de_Di2_known
                _, indices = torch.sort(u_flat, descending=False)
                k = max(1, int(len(u_flat) * ratio_Di2_known))              ### 至少保留1个
                topk_indices_in_known_2 = indices[:k]
                known_indices_2 = torch.nonzero(known_mask, as_tuple=True)[0]
                selected_indices_2 = known_indices_2[topk_indices_in_known_2]
                new_known_mask_2 = torch.zeros_like(known_mask, dtype=torch.bool)
                new_known_mask_2[selected_indices_2] = True
                known_mask = new_known_mask_2                     ###使用new_known_mask替代known_mask 


                rejection_mask = known_mask | unknown_mask      ###update also rejection_mask


                ###获取新unknown和known
                pseudo_labels_unknow = pseudo_labels[unknown_mask]                       ###
                y_hat_Di1_unknow  = y_hat_Di1[unknown_mask]                              ###
                y_unknow = y[unknown_mask]                                               ###
                de_Di2_unknow = de_Di2[unknown_mask]                                     ###
                u_Di1_unknow = u_Di1[unknown_mask]                                       ###
                
                pseudo_labels_know = pseudo_labels[known_mask]                           ###
                y_hat_Di1_know  = y_hat_Di1[known_mask]                                  ###
                y_know = y[known_mask]                                                   ###
                de_Di2_know = de_Di2[known_mask]                                         ###
                u_Di1_know = u_Di1[known_mask]                                           ###
                

                ###打印新标签
                #entropy_values_test_unknow = entropy_values_test[unknown_mask] 
                #entropy_values_test_unknow = torch.tensor(entropy_values_test_unknow)
                #entropy_values_test_know = entropy_values_test[known_mask] 
                #entropy_values_test_know = torch.tensor(entropy_values_test_know)
                print_sorted(pseudo_labels_unknow,y_hat_Di1_unknow,y_unknow, de_Di2_unknow, u_Di1_unknow,descending=True,name="final pseudo_labels_unknow, y_hat_Di1_unknow, y_unknow, de_Di2_unknow, u_Di1_unknow")
                print_sorted(pseudo_labels_know,y_hat_Di1_know,y_know, de_Di2_know,u_Di1_know,descending=True,name="final pseudo_labels_know, y_hat_Di1_know, y_know, de_Di2_know, u_Di1_know")


                ###打印新伪标签精度
                final_correct_unknow = (pseudo_labels_unknow == y_unknow).sum().item()
                final_correct_know = (pseudo_labels_know == y_know).sum().item()

                final_total_unknow = y_unknow.numel()
                final_total_know = y_know.numel()

                final_acc_pseudo_labels_unknow = final_correct_unknow / final_total_unknow if final_total_unknow > 0 else 0.0
                final_acc_pseudo_labels_know = final_correct_know / final_total_know if final_total_know > 0 else 0.0

                self.final_num_correct_unknow_accum += final_correct_unknow
                self.final_num_correct_know_accum += final_correct_know
                self.final_num_all_unknow_accum += final_total_unknow
                self.final_num_all_know_accum += final_total_know

                final_acc_pseudo_labels_unknow_accum = self.final_num_correct_unknow_accum / self.final_num_all_unknow_accum if self.final_num_all_unknow_accum > 0 else 0.0
                final_acc_pseudo_labels_know_accum = self.final_num_correct_know_accum / self.final_num_all_know_accum if self.final_num_all_know_accum > 0 else 0.0

                print(f"\nfinal_acc_pseudo_labels_unknow_step: {final_acc_pseudo_labels_unknow}")
                print(f"final_acc_pseudo_labels_know_step: {final_acc_pseudo_labels_know}")
                print(f"final_acc_pseudo_labels_unknow_accum: {final_acc_pseudo_labels_unknow_accum}")
                print(f"final_acc_pseudo_labels_know_accum: {final_acc_pseudo_labels_know_accum}")


            ###test pseudo_labels
            print(f"pseudo_labels: {pseudo_labels}")                               ### 
            print(f"y_ground truth: {y}")                                          ###
            pseudo_labels_same_ratio = (pseudo_labels == y).float().mean().item()  ###
            print("pseudo_labels same_ratio pseudo_labels/y_ground truth:", pseudo_labels_same_ratio * 100, "%")   ###
                                 ###

            

            ###打印(最终)伪标签准确率
            # unknown
            num_true_unknown_mask = true_unknown_mask.sum().item()            ### 
            if num_true_unknown_mask > 0:
                unk_same_ratio = (((pseudo_labels == y) & true_unknown_mask).sum().item()) / num_true_unknown_mask
            else:
                unk_same_ratio = 0.0
            # known 
            num_true_known_mask = true_known_mask.sum().item()            ### 
            if num_true_known_mask > 0:
                kn_same_ratio = (((pseudo_labels == y) & true_known_mask).sum().item()) / num_true_known_mask
            else:
                kn_same_ratio = 0.0

            print("pseudo_labels Unknown same_ratio pres/true:", unk_same_ratio * 100, "%")           ### correct pseudo_labels/true
            print("pseudo_labels not Unknown same_ratio pres/true:", kn_same_ratio * 100, "%")        ### correct pseudo_labels/true


            ###y_hat Accuracy
            #print(f"===============y_hat_Di===============: {y_hat_Di}")          ####
            #print(f"==============y_true_hat==============: {y}")                 ####
            #y_hat_Di1_known = y_hat_Di1[true_known_mask]
            #y_true_hat_known = y[true_known_mask]
            #y_hat_Di1_known_accuracy = (y_hat_Di1_known == y_true_hat_known).sum().float() / true_known_mask.sum()
            #print(f"==========y_hat_Di_accuracy==========: {y_hat_Di_known_accuracy}") 
            #print("==========uncertainty u,y_hat_Di,y_true_hat,pseudo_labels==========: ") 
            #print(torch.cat([u_Di, y_hat_Di.unsqueeze(1).float(), y.unsqueeze(1).float(),pseudo_labels.unsqueeze(1).float()], dim=1))
            #print(f"--------uncertainty u,y_hat_Di,y_true_hat--------: {u_Di},{y_hat_Di},{y}")


            if self.pseudo_label_quality == 1.0:                                 ###若采用真实标签，用真实标签和mask替代
                pseudo_labels = y.clone()                                        ###
                known_mask = true_known_mask                                     ###
                unknown_mask = true_unknown_mask                                 ###
                rejection_mask = true_rejection_mask                             ###
                print(f"pseudo_labels: {pseudo_labels}")                         ### 
                print(f"y_ground truth: {y}")                                    ###


            # ---------- Enable OPDA for predictions ----------
            if self.Dirichlet == 1.0:
                self.annealing_pres_step = 300                  ### old 100
                annealing_pres_coef = min(1.0, self.global_step / self.annealing_pres_step)
                _, preds = torch.max(y_hat_clone_detached, dim=1)
                unknown_threshold = 0.7 * ((self.mask.tau_low + self.mask.tau_high) / 2) * (annealing_pres_coef)  +  0.01 * ((self.mask.tau_low + self.mask.tau_high) / 2)     ###threshold for determining unknown labels
            else:
                _, preds = torch.max(y_hat_clone_detached, dim=1)
                unknown_threshold = (self.mask.tau_low + self.mask.tau_high) / 2


            entropy_values = calculate_entropy(likelihood)
            output_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
            output_mask[entropy_values >= unknown_threshold] = True
            preds[output_mask] = self.source_class_num

            print(f"y_ground truth: {y}")                                     ###
            print(f"preds: {preds}")                                          ###
            preds__same_ratio = (preds == y).float().mean().item()            ###
            print("preds same_ratio preds/y_ground truth:", preds__same_ratio * 100, "%") 

            ###打印(最终)预测准确率
            # unknown
            num_true_unknown_mask = true_unknown_mask.sum().item()            ### 
            if num_true_unknown_mask > 0:
                unk_same_ratio = (((preds == y) & true_unknown_mask).sum().item()) / num_true_unknown_mask
            else:
                unk_same_ratio = 0.0
            # known 
            num_true_known_mask = true_known_mask.sum().item()                ### 
            if num_true_known_mask > 0:
                kn_same_ratio = (((preds == y) & true_known_mask).sum().item()) / num_true_known_mask
            else:
                kn_same_ratio = 0.0

            print("preds Unknown same_ratio pres/true:", unk_same_ratio * 100, "%")           ### correct pres/true
            print("preds not Unknown same_ratio pres/true:", kn_same_ratio * 100, "%")        ### correct pres/true

            num_pres_unknown = (preds == self.source_class_num).sum()
            print(f"estimated number of unknown labels:  estimated unknown labels /total true labels: {num_pres_unknown}/64")

            print(f"number of true unknown labels:  true unknown labels /total true labels: {num_true_unknown_mask}/64")

            print(f"--------------self.tau_low: {self.mask.tau_low}")             ###test
            print(f"-------------self.tau_high: {self.mask.tau_high}")            ###test

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
            logits1 = logits1.to(self.device)                 # logits
            feat_ext_aug = feat_ext_aug.to(self.device)
            logits1_aug = logits1_aug.to(self.device)         # logits_aug
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
            if self.Dirichlet == 1.0:
                ### ---------- Dirichlet-evidence loss ----------        #####
                ###1.Dirichlet squares loss

                pseudo_labels_know = pseudo_labels[known_mask]
                one_hot_know = torch.nn.functional.one_hot(pseudo_labels_know, num_classes=self.source_class_num)
                one_hot_know_Di1 = one_hot_know.detach().float()
                alpha_Di1_known = alpha_Di1[known_mask].float()
                s_Di1_known = s_Di1[known_mask].float()
                per_sample_L_Di1_sl = torch.sum((one_hot_know_Di1 - alpha_Di1_known / s_Di1_known) ** 2 + alpha_Di1_known * (s_Di1_known - alpha_Di1_known) / (s_Di1_known ** 2 * (s_Di1_known + 1)), dim=1 )
                L_Di1_sl = torch.mean(per_sample_L_Di1_sl)    ###14.04.2026


                ###2.Dirichlet  KL divergence loss
                alpha_Di1_unknow = alpha_Di1[unknown_mask]       ###14.04.2026
                alpha_Di1_tilde = alpha_Di1_unknow             ###因为unknown样本含有的全是错误alpha，所以期望其全部向1靠近
                #alpha_Di1_tilde = one_hot_know_Di1 + (1 - one_hot_know_Di1)*alpha_Di
                per_sample_L_Di1_kl = kl_dirichlet(alpha_Di1_tilde,self.source_class_num)
                L_Di1_kl = torch.mean(per_sample_L_Di1_kl)    ###14.04.2026

                self.annealing_step = 100                  ### old 200
                annealing_coef = min(1.0, self.global_step / self.annealing_step)
                a_L_Di1_kl = annealing_coef * L_Di1_kl

                #print(f"\nself.global_step:{self.global_step}") 
                #print(f"L_Di2: {L_Di2}") 
                #print(f"annealing_coef: {annealing_coef}") 
                #print(f"a_L_Di2: {a_L_Di2}")

                log_metrics(
                    {

                    "acc/final_step_unknow_pseudo_labels acc": final_acc_pseudo_labels_unknow,
                    "acc/final_step_know_pseudo_labels acc": final_acc_pseudo_labels_know,
                    "acc/final_accumulated_unknow_pseudo_labels acc": final_acc_pseudo_labels_unknow_accum,
                    "acc/final_accumulated_know_pseudo_labels acc": final_acc_pseudo_labels_know_accum,

                    "acc/step_total_pseudo_labels_acc": pseudo_labels_same_ratio,     ###correct/true
                    "acc/selected_step_unknow_pseudo_labels_acc": acc_pseudo_labels_unknow,     ###correct/true
                    "acc/selected_step_know_pseudo_labels_acc": acc_pseudo_labels_know,     ###correct/true
                    "acc/selected_accumulated_unknow_pseudo_labels_acc": acc_pseudo_labels_unknow_accum,     ###correct/true
                    "acc/selected_accumulated_know_pseudo_labels_acc": acc_pseudo_labels_know_accum,     ###correct/true

                    "acc/step_total_preds_acc": preds__same_ratio,
                    "acc/step_unknow_preds_acc": unk_same_ratio,
                    "acc/step_know_preds_acc": kn_same_ratio,

                    "num_preds_unknown/64": num_pres_unknown,
                    "num_true_unknown/64": num_true_unknown_mask,


                    "tau_low": self.mask.tau_low,
                    "tau_high": self.mask.tau_high,

                    },
                    step=self.global_step,
                    prefix="adaptation"
                )
            else:
                log_metrics(
                    {

                    "acc/step_total_pseudo_labels_acc": pseudo_labels_same_ratio,     ###correct/true
                    "acc/selected_step_unknow_pseudo_labels_acc": acc_pseudo_labels_unknow,     ###correct/true
                    "acc/selected_step_know_pseudo_labels_acc": acc_pseudo_labels_know,     ###correct/true
                    "acc/selected_accumulated_unknow_pseudo_labels_acc": acc_pseudo_labels_unknow_accum,     ###correct/true
                    "acc/selected_accumulated_know_pseudo_labels_acc": acc_pseudo_labels_know_accum,     ###correct/true

                    "acc/step_total_preds_acc": preds__same_ratio,
                    "acc/step_unknow_preds_acc": unk_same_ratio,
                    "acc/step_know_preds_acc": kn_same_ratio,

                    "num_preds_unknown/64": num_pres_unknown,
                    "num_true_unknown/64": num_true_unknown_mask,


                    "tau_low": self.mask.tau_low,
                    "tau_high": self.mask.tau_high,

                    },
                    step=self.global_step,
                    prefix="adaptation"
                )

            if self.Dirichlet == 1.0:
                #self.loss = L_con + self.lam * L_kl   ###old    loss function
                #self.loss = - L_con + self.lam * L_kl  + 1000000*L_Di1_sl + 1000000*a_L_Di1_kl
                #self.loss = - L_con + self.lam * L_kl  + 100*L_Di1_sl + 100*a_L_Di1_kl
                #self.loss = - L_con + self.lam * L_kl  + 10*L_Di1_sl + 10*a_L_Di1_kl
                #self.loss = - L_con + self.lam * L_kl  + 5*L_Di1_sl + 5*a_L_Di1_kl
                #self.loss = - L_con + self.lam * L_kl  + 2*L_Di1_sl + 2*a_L_Di1_kl
                self.loss = - L_con + self.lam * L_kl  + L_Di1_sl + a_L_Di1_kl
                #self.loss = - L_con + self.lam * L_kl  + 0.1*L_Di1_sl + 0.1*a_L_Di1_kl  ###loss function
                #self.loss = L_Di1_sl + a_L_Di1_kl
                #self.loss = 0.5*L_Di1_sl + 0.5*a_L_Di1_kl
                #self.loss = 0.1*L_Di1_sl + 0.1*a_L_Di1_kl
                #self.loss = 0.05*L_Di1_sl + 0.05*a_L_Di1_kl
            
                print(f"self.loss: {self.loss.item()}")                              ###
                print(f"L_con: {L_con.item()}, L_kl: {L_kl.item()}")                 ###
                print(f"L_Di_sl: {L_Di1_sl.item()}, L_Di1_kl: {L_Di1_kl.item()}")    ###, L_Di2: {L_Di2.item()}
                
            else:
                self.loss = - L_con + self.lam * L_kl

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
