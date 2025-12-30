import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from torch.distributions import Normal 

class MultivariateNormalDiag():
    
    def __init__(self, locs, scales):
        super(MultivariateNormalDiag, self).__init__()
        self.locs = locs
        self.scales = scales

    def log_prob(self, x):

        normal = Normal(loc=self.locs, scale=self.scales)
        return normal.log_prob(x).sum(-1)
    
    def sample(self, shape=()):
        eps = torch.randn(shape + (self.locs.shape[1],)).cuda() 
        return self.locs + self.scales * eps
  


def disentML(input, input_bg,input_specific, proxy_l2, target, scale, num_classes):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] labels

    '''
    proxy_l2 = F.normalize(proxy_l2, p=2, dim=1)
    dim=int(input.shape[1]/2)
    mu = input[:,:dim]
    logvar = input[:,dim:]
    std = F.softplus(logvar-5)
    z_dist = MultivariateNormalDiag(mu, std)
    
    input_rdn = z_dist.sample ((mu.shape[0],))
    input_rdn_l2 = F.normalize(input_rdn, p=2, dim=1)
    
    
  
    

    
    
# for z loss
    sim_mat = input_rdn_l2.matmul(proxy_l2.t())
        
    logits = scale * sim_mat
    one_hot = F.one_hot(target, num_classes)
    neg_hot = 1.0 - one_hot                                         
    neg_target = neg_hot / (neg_hot.sum(dim=1, keepdim=True) + 1e-12)  

    
    log_probs = F.log_softmax(logits, dim=1)        # logits: [B, C]
    agnostic_loss = -(neg_target * log_probs).sum(dim=1).mean()

# for zs loss
    zs_mu = input_specific[:,:dim]
    zs_logvar = input_specific[:,dim:]
    
    zs_std = F.softplus(zs_logvar-5,beta=1)

    z_s_dist = MultivariateNormalDiag(zs_mu, zs_std)
    input_spcf_rdn = z_s_dist.sample ((zs_mu.shape[0],))

    input_spcf_l2_rdn = F.normalize(input_spcf_rdn, p=2, dim=1)
    sim_mat_spcf = input_spcf_l2_rdn.matmul(proxy_l2.t())     
    logits_spcf = scale * sim_mat_spcf
        
    specific_loss = F.cross_entropy(logits_spcf, target)

    zs_var = zs_std.pow(2)
    zs_kl_prior_loss = 0.5 * (zs_mu.pow(2) + zs_var - torch.log(zs_var + 1e-12) - 1.0).mean()

# for z_bg loss
    bg_mu = input_bg[:,:dim]
    bg_logvar = input_bg[:,dim:]
    
    bg_std = F.softplus(bg_logvar-5,beta=1)
    
    z_bg_dist = MultivariateNormalDiag(bg_mu, bg_std)
    input_bg_rdn = z_bg_dist.sample ((bg_mu.shape[0],))

    input_bg_l2_rdn = F.normalize(input_bg_rdn, p=2, dim=1)
    sim_mat_bg = input_bg_l2_rdn.matmul(proxy_l2.t())     
    logits_bg = scale * sim_mat_bg    

    log_probs = F.log_softmax(logits_bg, dim=1)        # logits: [B, C]
    bg_loss = -(neg_target * log_probs).sum(dim=1).mean()

    bg_var = bg_std.pow(2)
    bg_kl_prior_loss = 0.5 * (bg_mu.pow(2) + bg_var - torch.log(bg_var + 1e-12) - 1.0).mean()

    # ----- HSIC split: (mu,std) vs (bg_mu,bg_std) -----
    z_feat   = torch.cat([mu, std],    dim=1)   # [B, 2D]
    zbg_feat = torch.cat([bg_mu, bg_std], dim=1)   # [B, 2D]

    # bandwidth는 None이면 median heuristic 사용됨
    # normalize=True 권장(스케일 민감도 감소)
    split_loss = hsic_rbf(z_feat, zbg_feat, sigma_x=None, sigma_y=None, normalize=True)

# z_bg_ent

    tau=1
    logp_bg = F.log_softmax(logits_bg / tau, dim=1)
    uniform = torch.full_like(logp_bg.exp(), 1.0 / num_classes)
    bg_ent_loss = F.kl_div(logp_bg, uniform, reduction='batchmean')  # = -H(p)+const      

# === New: repulsion against the true class ===
    repulsion_weight = 1.0   # 튜닝용 하이퍼 (아래 Proxy_Anchor에 파라미터로 빼도 됨)
    pos_logp = (log_probs * one_hot).sum(dim=1)  # = log p(y|z_bg)
    bg_repulsion_loss = pos_logp.mean()          # minimize → p(y)↓ (정답에서 멀어짐)

##z_bg <->proxy ortho
    P_y = proxy_l2[target]                           # [B, D], 각 샘플의 정답 프록시
    cos_y = (input_bg_l2_rdn * P_y).sum(dim=1)  # [B], z_bg vs true proxy cosine

    # margin m 이하로 유지: cos_y > m 이면 페널티
    m = 0.0  # 0이면 직교 이상으로 벌리기; 살짝 음수(-0.05~-0.1)로 두면 더 강한 repel
    bg_prx_ortho_loss = F.relu(cos_y - m).mean()


#ortho

    # --- orthogonal split loss (decorrelation between (mu,std) and (rel_mu,rel_std)) ---
    # batch-center
    z_feat     = z_feat - z_feat.mean(dim=0, keepdim=True)
    zbg_feat  = zbg_feat - zbg_feat.mean(dim=0, keepdim=True)    
    # focus on directions
    z_feat     = F.normalize(z_feat, p=2, dim=1)
    zbg_feat  = F.normalize(zbg_feat, p=2, dim=1)    
    # --- orthogonal split loss (decorrelation between (mu,std) and (rel_mu,rel_std)) ---
    z_feat    = torch.cat([mu, std], dim=1)          # [B, 2D]
    zbg_feat  = torch.cat([bg_mu, bg_std], dim=1)  # [B, 2D]
    # cross-covariance & Frobenius norm → encourage orthogonality
    B_ortho = z_feat.size(0)
    C_ortho = (z_feat.transpose(0, 1) @ zbg_feat) / (max(B_ortho - 1, 1))
    ortho_loss = (C_ortho ** 2).sum()






    return agnostic_loss, specific_loss, zs_kl_prior_loss, bg_loss, bg_kl_prior_loss, split_loss, logits, logits_bg, logits_spcf, ortho_loss, bg_ent_loss, bg_repulsion_loss, bg_prx_ortho_loss





def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    # K: [B, B]
    B = K.size(0)
    K_mean_row = K.mean(dim=0, keepdim=True)      # [1,B]
    K_mean_col = K.mean(dim=1, keepdim=True)      # [B,1]
    K_mean_all = K.mean()                          # scalar
    return K - K_mean_row - K_mean_col + K_mean_all

def _rbf_kernel(X, sigma=None, eps=1e-12):
    pd2 = torch.cdist(X, X) ** 2  # [B,B]
    if sigma is None:
        B = X.size(0)
        if B > 1:
            off = ~torch.eye(B, dtype=torch.bool, device=X.device)
            med = torch.median(pd2.detach()[off])
        else:
            med = pd2.new_tensor(1.0)  # 임시값
        sigma2 = torch.clamp(med, min=eps)
    else:
        sigma2 = torch.clamp(sigma**2, min=eps)
    return torch.exp(-pd2 / (2.0 * sigma2))


def hsic_rbf(X: torch.Tensor, Y: torch.Tensor, sigma_x: torch.Tensor = None, sigma_y: torch.Tensor = None,
             normalize: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """
    HSIC(X,Y) = tr(Kc Lc) / (B-1)^2
    - X: [B,Dx], Y: [B,Dy]
    - sigma_x, sigma_y: (선택) X,Y용 RBF bandwidth. 주지 않으면 median heuristic 사용.
    - normalize=True면 ||Kc||_F ||Lc||_F 로 나눠 스케일 민감도 감소.
    """
    B = X.size(0)
    if B < 2:
        return X.new_zeros(())

    K = _rbf_kernel(X, sigma=sigma_x, eps=eps)
    L = _rbf_kernel(Y, sigma=sigma_y, eps=eps)

    Kc = _center_kernel(K)
    Lc = _center_kernel(L)

    hsic_val = (Kc * Lc).sum() / ((B - 1) ** 2)

    if normalize:
        denom = (Kc.pow(2).sum().clamp_min(eps).sqrt() *
                 Lc.pow(2).sum().clamp_min(eps).sqrt())
        hsic_val = hsic_val / denom

    return hsic_val

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, al=0.0, be=0.0, gam=0.0, de = 0.0, eps = 0.0, ze = 0.0, eta= 0.0, theta= 0.0, iota=0.0, kap=0.0):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.scale = 23.0

        
        self.al = al
        self.be = be
        self.gam = gam
        self.de = de
        self.eps = eps
        self.ze = ze
        self.eta= eta
        self.theta= theta
        self.iota= iota
        self.kap= kap
        
    def forward(self, X, x_bg, X_specific, T, disent):
        if disent:
            P = self.proxies
            agnostic_loss, specific_loss, zs_kl_prior_loss, bg_loss, bg_kl_prior_loss, split_loss, logits, logits_bg, logits_spcf,ortho_loss, bg_ent_loss, bg_repulsion_loss, bg_prx_ortho_loss = disentML(X, x_bg, X_specific, l2_norm(P), T, self.scale, num_classes=self.nb_classes)
            AGReg =  self.al *agnostic_loss + self.be*specific_loss + self.gam*zs_kl_prior_loss + self.de*bg_loss + self.eps*bg_kl_prior_loss + self.ze*split_loss+self.eta*ortho_loss +self.theta*bg_ent_loss + self.iota*bg_repulsion_loss + self.kap*bg_prx_ortho_loss

            emb_mu = X[:,:self.sz_embed]
            input_rdn = emb_mu
            input_rdn_l2 = F.normalize(input_rdn, p=2, dim=1)

            cos = F.linear(input_rdn_l2, l2_norm(P))  # Calcluate cosine similarity
            P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
            N_one_hot = 1 - P_one_hot
        
            pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
            neg_exp = torch.exp(self.alpha * (cos + self.mrg))

            with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # set of positive proxies of data in  batch
            num_valid_proxies = len(with_pos_proxies)   # number of positive proxies
            
            P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
            N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
            
            pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
            neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
            dml_loss = pos_term + neg_term   

            loss = AGReg + dml_loss

            return loss, logits, logits_bg, logits_spcf, [ agnostic_loss, specific_loss, zs_kl_prior_loss, bg_loss, bg_kl_prior_loss, split_loss, ortho_loss, bg_ent_loss, bg_repulsion_loss, bg_prx_ortho_loss ]

        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   #  set of positive proxies of data in  batch
        num_valid_proxies = len(with_pos_proxies)   #  number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss