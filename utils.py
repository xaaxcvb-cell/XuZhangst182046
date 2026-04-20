import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import math
from scipy.stats import multivariate_normal

import random                      ###16.04
from collections import deque      ###16.04

import wandb                       ###18.04


def log_metrics(metrics: dict, step: int, prefix: str = None):
    """
    Log multiple metrics to wandb.

    Args:
        metrics (dict): e.g. {"loss": 0.1, "acc": 0.95}
        step (int): global step
        prefix (str, optional): e.g. "train", "val"
    """
    log_dict = {}
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = v.item()
        name = f"{prefix}/{k}" if prefix else k
        log_dict[name] = v

    wandb.log(log_dict, step=step)


def calculate_entropy(likelihood):
    entropy_values = -(likelihood * torch.log2(likelihood + 1e-10)).sum(dim=1)
    scale_factor = torch.log2(torch.tensor(likelihood.shape[1]))
    entropy_values = entropy_values / scale_factor
    return entropy_values


def calculate_kld(likelihood, true_dist):
    T = 0.1
    dividend = torch.sum(torch.exp(likelihood / T), dim=1)
    logarithmus = - torch.log(dividend)
    divisor = torch.sum(true_dist, dim=1)
    kld_values = - (1 / likelihood.shape[1]) * divisor * logarithmus
    return kld_values


def calculate_cosine_similarity(mu, feat):
    cosine_sim = F.cosine_similarity(mu.unsqueeze(0), feat.unsqueeze(1), dim=2)
    return cosine_sim

###new dirichlet D2
def kl_dirichlet(alpha, K):
    beta = torch.ones((1, K), device=alpha.device, dtype=alpha.dtype)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


###old dirichlet D2
#def kl_dirichlet(alpha_tilde):  ###~~~~~~~~~~~~~~
    """
    alpha_tilde: [B, K]
    return: [B]
    """
    B, K = alpha_tilde.shape

    sum_alpha = alpha_tilde.sum(dim=1, keepdim=True)      #

    term1 = (
        torch.lgamma(sum_alpha)                           #
        - torch.lgamma(torch.tensor(float(K), device=alpha_tilde.device))  #
        - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)               #
    )

    term2 = ((alpha_tilde - 1.0) *                        # 
             (torch.digamma(alpha_tilde)                  # 
              - torch.digamma(sum_alpha))                 # 
            ).sum(dim=1, keepdim=True)                    # 

    kl = (term1 + term2).squeeze(1)                       # 
    return kl


###Differential Entropy of Dirichlet Prior Network
def DE_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.to(dtype=torch.float64)

    if alpha.ndim == 1:
        alpha0 = torch.sum(alpha)
        term1 = torch.sum(torch.lgamma(alpha))    ###lgamma就是ln gamma
        term2 = torch.lgamma(alpha0)
        term3 = torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0)))
        H = term1 - term2 - term3
    else:
        alpha0 = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]  dim=1，在K上求和
        term1 = torch.sum(torch.lgamma(alpha), dim=1)   # [B]  dim=1，在K上求和
        term2 = torch.lgamma(alpha0).squeeze(1)         # [B]
        term3 = torch.sum(
            (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha0)),
            dim=1
        )                                               # [B]
        H = term1 - term2 - term3

    return H


def print_sorted(pseudo_labels, y_hat_Di, y, u_Di, de_Di, descending, name):

    title = f"\n=========={name}=========="
    print(title)

    data = torch.cat([
        pseudo_labels.unsqueeze(1).float(),
        y_hat_Di.unsqueeze(1).float(),
        y.unsqueeze(1).float(),
        u_Di.unsqueeze(1).float(),
        de_Di.unsqueeze(1).float()
    ], dim=1)

    _, indices = torch.sort(data[:, -1], descending=descending)
    data_sorted = data[indices]

    print(data_sorted)

    return data_sorted


class mask():
    def __init__(self, known_percentage_threshold, unknown_percentage_threshold, N_init):
        self.known_percentage_threshold = known_percentage_threshold
        self.unknown_percentage_threshold = unknown_percentage_threshold

        self.tau_low = None
        self.tau_low_list = []
        self.tau_high = None
        self.tau_high_list = []

        self.count = 0
        self.N_init = N_init

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        ###
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_mask(self, likelihood):
        entropy_values = calculate_entropy(likelihood)
        if self.count < self.N_init:
            # Sort values (from small to big)

            print(f"--------------self.known_percentage_threshold: {self.known_percentage_threshold}")            ###
            print(f"-------------self.unknown_percentage_threshold: {self.unknown_percentage_threshold}")            ###

            sorted_A, _ = torch.sort(entropy_values)
            threshold_idx_known = math.ceil(len(sorted_A) * (self.known_percentage_threshold))
            threshold_a = sorted_A[threshold_idx_known]
            self.tau_low_list.append(threshold_a)
            tau_low = torch.tensor(self.tau_low_list)
            threshold_idx_unknown = math.floor(len(sorted_A) * (self.unknown_percentage_threshold))
            threshold_b = sorted_A[threshold_idx_unknown]
            self.tau_high_list.append(threshold_b)
            tau_high = torch.tensor(self.tau_high_list)
            self.tau_low = torch.mean(tau_low)
            self.tau_high = torch.mean(tau_high)

            print(f"--------------self.tau_low: {self.tau_low}")            ###
            print(f"-------------self.tau_high: {self.tau_high}")            ###

            self.count = self.count + 1

        # Determine the threshold value for the percentage_threshold values
        known_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
        known_mask[entropy_values < self.tau_low] = True
        tau_low = self.tau_low
        while torch.sum(known_mask).item() <= 1:
            tau_low += self.tau_low
            known_mask = entropy_values <= tau_low   #~~~~ 实际上并没有小于等于1的情况

        unknown_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
        unknown_mask[entropy_values > self.tau_high] = True
        tau_high = self.tau_high
        while torch.sum(unknown_mask).item() <= 1:
            tau_high -= self.tau_high
            unknown_mask = entropy_values >= tau_high  #~~~~ 实际上并没有小于等于1的情况

        both_true = torch.logical_and(known_mask, unknown_mask)
        unknown_mask[both_true] = False

        rejection_mask = (known_mask | unknown_mask)

        return known_mask, unknown_mask, rejection_mask  #?bulsche?  ###rejection_mask: known_mask or unknown_mask 即都算


class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, iter_max):
        self.optimizer = optimizer
        self.iter_max = iter_max

        super(CustomLRScheduler, self).__init__(optimizer)

    # update optimizer
    def step(self, iter_num=0, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / self.iter_max) ** (-power)
        # for every parameter in the list
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return self.optimizer

"""
class GaussianMixtureModel():
    def __init__(self, source_class_num):
        self.source_class_num = source_class_num
        self.batch_weight = torch.zeros(source_class_num, dtype=torch.float)
        self.mu = None
        self.C = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    def soft_update(self, feat, posterior):
        # Set the desired data type
        dtype = torch.float64  # You can use torch.float128 if supported

        torch.set_printoptions(threshold=float('inf'))

        posterior = posterior.to(dtype)
     
        feat = feat.to(device=self.device, dtype=dtype)                               #old
        self.batch_weight = self.batch_weight.to(device=self.device, dtype=dtype)     #old

        # ---------- Calculate mu ----------
        # Calculate the sum of the posteriors
        batch_weight_new = torch.zeros(posterior.shape[1], device=self.device, dtype=dtype)
        batch_weight_new = batch_weight_new + torch.sum(posterior, dim=0)

        batch_weight_new = batch_weight_new + self.batch_weight

        # Calculate the sum of the weighted features
        weighted_sum = torch.matmul(posterior.T, feat)

        if self.mu != None:
            weighted_sum = torch.multiply(self.batch_weight.unsqueeze(1), self.mu) + weighted_sum

        # Calculate mu
        mu_new = weighted_sum / batch_weight_new[:, None]

        # ---------- Calculate the Covariance Matrices ----------
        # Calculate the sum of the outer product
        differences = feat.unsqueeze(1) - mu_new.unsqueeze(0)

        outer_prods = torch.einsum('nmd,nme->nmde', differences, differences)

        epsilon = 1e-6
        eye = torch.eye(differences.shape[2], device=self.device).unsqueeze(0).unsqueeze(0)
        outer_prods = 0.5 * (outer_prods + outer_prods.transpose(-1, -2)) + epsilon * eye

        posterior_expanded = posterior.unsqueeze(-1).unsqueeze(-1)
        weighted_sum = torch.sum(posterior_expanded * outer_prods, dim=0)

        if self.C != None:
            weighted_sum = self.C * self.batch_weight.unsqueeze(1).unsqueeze(2) + weighted_sum

        # Calculate C
        C_new = weighted_sum / batch_weight_new[:, None, None]

        self.batch_weight = batch_weight_new
        self.mu = mu_new
        self.C = C_new

    def get_likelihood(self, feat, mu, C):
        torch.set_printoptions(threshold=float('inf'))
        likelihood = torch.zeros((mu.shape[0], feat.shape[0]))

        # Compute the likelihood of the features for each class
        for i, (mean, cov) in enumerate(zip(mu, C)):
            mean = mean.cpu().detach().numpy() if isinstance(mean, torch.Tensor) else mean
            cov = cov.cpu().detach().numpy() if isinstance(cov, torch.Tensor) else cov
            feat = feat.cpu() if feat.is_cuda else feat
            rv = multivariate_normal(mean, cov, allow_singular=True)
            likelihood[i, :] = torch.from_numpy(rv.logpdf(feat)).type_as(likelihood)

        # for numerical stability
        maximum_likelihood = torch.max(likelihood).item()
        likelihood = likelihood - maximum_likelihood  #所有likelihood都减去maximum_likelihood。
        likelihood = torch.exp(likelihood)

        # Normalize the likelihood
        likelihood = likelihood / torch.sum(likelihood, axis=0, keepdims=True)

        likelihood = likelihood.T

        return likelihood

    def get_labels(self, feat):                                        ####
        likelihood = self.get_likelihood(feat, self.mu, self.C)
        max_values, max_indices = torch.max(likelihood, dim=1)

        return max_values, max_indices, likelihood
"""

class GaussianMixtureModel():
    """
    Keep the same class name / interfaces, but internally use Student-t Mixture Model (TMM).
    External methods unchanged:
        - soft_update(feat, posterior)
        - get_likelihood(feat, mu, C)
        - get_labels(feat)
    """
    def __init__(self, source_class_num, nu=10.0, eps=1e-6):
        self.source_class_num = source_class_num
        self.batch_weight = torch.zeros(source_class_num, dtype=torch.float64)
        self.mu = None
        self.C = None

        # degrees of freedom for each component in TMM
        # keep fixed to avoid changing outer logic
        self.nu = torch.full((source_class_num,), float(nu), dtype=torch.float64)

        self.eps = eps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _regularize_cov(self, C):
        """
        Ensure covariance is symmetric positive semi-definite and numerically stable.
        C: [K, D, D]
        """
        K, D, _ = C.shape
        eye = torch.eye(D, device=C.device, dtype=C.dtype).unsqueeze(0)
        C = 0.5 * (C + C.transpose(-1, -2))
        C = C + self.eps * eye
        return C

    def _mahalanobis_squared(self, feat, mu, C):
        """
        Compute squared Mahalanobis distance:
            delta[n, k] = (x_n - mu_k)^T C_k^{-1} (x_n - mu_k)
        feat: [N, D]
        mu:   [K, D]
        C:    [K, D, D]
        return: [N, K]
        """
        feat = feat.to(self.device)
        mu = mu.to(self.device)
        C = self._regularize_cov(C.to(self.device))

        N, D = feat.shape
        K = mu.shape[0]

        diff = feat.unsqueeze(1) - mu.unsqueeze(0)   # [N, K, D]

        # use pseudo-inverse for stability
        C_inv = torch.linalg.pinv(C)                 # [K, D, D]

        # delta[n,k] = diff[n,k,:]^T C_inv[k] diff[n,k,:]
        delta = torch.einsum('nkd,kde,nke->nk', diff, C_inv, diff)
        delta = torch.clamp(delta, min=0.0)
        return delta

    def _student_t_logpdf(self, feat, mu, C, nu):
        """
        Multivariate Student-t logpdf for each component.
        feat: [N, D]
        mu:   [K, D]
        C:    [K, D, D]
        nu:   [K]
        return: [N, K]
        """
        feat = feat.to(device=self.device, dtype=torch.float64)
        mu = mu.to(device=self.device, dtype=torch.float64)
        C = self._regularize_cov(C.to(device=self.device, dtype=torch.float64))
        nu = nu.to(device=self.device, dtype=torch.float64)

        N, D = feat.shape
        K = mu.shape[0]

        delta = self._mahalanobis_squared(feat, mu, C)   # [N, K]

        sign, logdet = torch.linalg.slogdet(C)           # [K]
        # if sign <= 0, covariance is problematic; eps regularization above usually avoids that
        logdet = torch.where(sign > 0, logdet, torch.full_like(logdet, 1e10))

        # log normalization constant
        # lgamma((nu + D)/2) - lgamma(nu/2)
        # - 0.5 * (D log(nu*pi) + logdet)
        term1 = torch.lgamma((nu + D) / 2.0) - torch.lgamma(nu / 2.0)   # [K]
        term2 = 0.5 * (D * torch.log(nu * math.pi) + logdet)            # [K]

        # - ((nu + D)/2) * log(1 + delta/nu)
        term3 = ((nu + D) / 2.0).unsqueeze(0) * torch.log1p(delta / nu.unsqueeze(0))  # [N, K]

        logpdf = term1.unsqueeze(0) - term2.unsqueeze(0) - term3        # [N, K]
        return logpdf

    def soft_update(self, feat, posterior):
        """
        Keep the same signature.
        posterior: [N, K]
        Internally use TMM robust reweighting:
            r_tilde[n,k] = posterior[n,k] * u[n,k]
            u[n,k] = (nu_k + D) / (nu_k + delta[n,k])
        """
        dtype = torch.float64
        feat = feat.to(device=self.device, dtype=dtype)
        posterior = posterior.to(device=self.device, dtype=dtype)
        self.batch_weight = self.batch_weight.to(device=self.device, dtype=dtype)
        self.nu = self.nu.to(device=self.device, dtype=dtype)

        N, D = feat.shape
        K = posterior.shape[1]

        # ---------- Initialization: first update can use GMM-style stats ----------
        # because TMM needs an initial mu/C to compute Mahalanobis distance
        if self.mu is None or self.C is None:
            batch_weight_new = torch.sum(posterior, dim=0) + self.batch_weight  # [K]

            weighted_sum = posterior.T @ feat                                   # [K, D]
            mu_new = weighted_sum / (batch_weight_new[:, None] + self.eps)

            diff = feat.unsqueeze(1) - mu_new.unsqueeze(0)                      # [N, K, D]
            outer = torch.einsum('nkd,nke->nkde', diff, diff)                   # [N, K, D, D]
            weighted_cov_sum = torch.sum(
                posterior.unsqueeze(-1).unsqueeze(-1) * outer, dim=0
            )                                                                   # [K, D, D]

            C_new = weighted_cov_sum / (batch_weight_new[:, None, None] + self.eps)
            C_new = self._regularize_cov(C_new)

            self.batch_weight = batch_weight_new
            self.mu = mu_new
            self.C = C_new
            return

        # ---------- TMM robust weights ----------
        delta = self._mahalanobis_squared(feat, self.mu, self.C)                # [N, K]
        u = (self.nu.unsqueeze(0) + D) / (self.nu.unsqueeze(0) + delta + self.eps)  # [N, K]

        # effective posterior
        posterior_eff = posterior * u                                           # [N, K]

        # ---------- Update mu ----------
        batch_weight_new = torch.sum(posterior_eff, dim=0) + self.batch_weight  # [K]

        weighted_sum = posterior_eff.T @ feat                                   # [K, D]
        if self.mu is not None:
            weighted_sum = self.batch_weight.unsqueeze(1) * self.mu + weighted_sum

        mu_new = weighted_sum / (batch_weight_new[:, None] + self.eps)

        # ---------- Update covariance ----------
        diff = feat.unsqueeze(1) - mu_new.unsqueeze(0)                          # [N, K, D]
        outer = torch.einsum('nkd,nke->nkde', diff, diff)                       # [N, K, D, D]

        weighted_cov_sum = torch.sum(
            posterior_eff.unsqueeze(-1).unsqueeze(-1) * outer, dim=0
        )                                                                       # [K, D, D]

        if self.C is not None:
            weighted_cov_sum = self.C * self.batch_weight.unsqueeze(1).unsqueeze(2) + weighted_cov_sum

        C_new = weighted_cov_sum / (batch_weight_new[:, None, None] + self.eps)
        C_new = self._regularize_cov(C_new)

        self.batch_weight = batch_weight_new
        self.mu = mu_new
        self.C = C_new

    def get_likelihood(self, feat, mu, C):
        """
        Keep the same interface and return shape.
        Return normalized component likelihoods (actually posterior-like normalized scores).
        """
        feat = feat.to(device=self.device, dtype=torch.float64)
        mu = mu.to(device=self.device, dtype=torch.float64)
        C = C.to(device=self.device, dtype=torch.float64)
        nu = self.nu.to(device=self.device, dtype=torch.float64)

        # [N, K]
        log_likelihood = self._student_t_logpdf(feat, mu, C, nu)

        # numerical stability
        max_per_sample = torch.max(log_likelihood, dim=1, keepdim=True)[0]
        likelihood = torch.exp(log_likelihood - max_per_sample)

        # normalize over components
        likelihood = likelihood / (torch.sum(likelihood, dim=1, keepdim=True) + self.eps)

        return likelihood

    def get_labels(self, feat):
        likelihood = self.get_likelihood(feat, self.mu, self.C)
        max_values, max_indices = torch.max(likelihood, dim=1)
        return max_values, max_indices, likelihood




class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)

        if inputs.shape != targets.shape:
            targets = torch.zeros_like(inputs).scatter(1, targets, 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


class HScore(torchmetrics.Metric):
    def __init__(self, known_classes_num, shared_classes_num):
        super(HScore, self).__init__()

        # Number of possible outcomes is total_classes_num
        self.total_classes_num = known_classes_num + 1
        self.shared_classes_num = shared_classes_num

        self.add_state("correct_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")
        self.add_state("total_per_class", default=torch.zeros(self.total_classes_num), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert preds.shape == target.shape
        # total_classes_num is the number of Source model outputs including the unknown class.
        for c in range(self.total_classes_num):
            self.total_per_class[c] = self.total_per_class[c] + (target == c).sum()
            self.correct_per_class[c] = self.correct_per_class[c] + ((preds == target) * (target == c)).sum()

    def compute(self):
        # Source-Private classes not included in known_acc
        per_class_acc = self.correct_per_class / (self.total_per_class + 1e-5)
        known_acc = per_class_acc[:self.shared_classes_num].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
        return h_score, known_acc, unknown_acc

###16.04
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)     ###构造一个长度capacity的双端数列作为buffer。self.buffer就是buffer本体。

    def __len__(self):
        return len(self.buffer)

    def push_batch(self, x: torch.Tensor, y: torch.Tensor): ###后续从buffer中取samples仍会更新当前网络梯度，但不会回传给buffer输入的来源
        x = x.detach().cpu()
        y = y.detach().cpu()
        for i in range(x.size(0)):
            self.buffer.append((x[i].clone(), y[i].clone()))

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size       ###return True  or  False

    def sample(self, batch_size: int, device=None):
        batch = random.sample(self.buffer, batch_size)         ###随机抽batch_size个tensor放到batch中
        bx = torch.stack([item[0] for item in batch], dim=0)   ###把batch中所有tensor存到列表item[0],再把item[0]转化为一个[B,原shape]的新tensor
        by = torch.stack([item[1] for item in batch], dim=0)

        if device is not None:
            bx = bx.to(device)
            by = by.to(device)
        return bx, by