import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TCL(nn.Module):
    def __init__(self, temperature=0.1, k1=5000.0, k2=1.0):

        super(TCL, self).__init__()
        self.temperature = temperature
        self.k1 = torch.tensor(k1,requires_grad=False)
        self.k2 = torch.tensor(k2,requires_grad=False)

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T)
        exp_dot_tempered = torch.exp((dot_product_tempered) / self.temperature)
        exp_dot_tempered_n = torch.exp(-1 * dot_product_tempered) 
       

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_positives = mask_similar_class * mask_anchor_out
        mask_negatives = ~mask_similar_class
        positives_per_samples = torch.sum(mask_positives, dim=1)
        negatives_per_samples = torch.sum(mask_negatives, dim=1)
        
        
        tcl_loss = torch.sum(-torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_positives, dim=1)+(self.k1*torch.sum(exp_dot_tempered_n * mask_positives, dim=1))+(self.k2*torch.sum(exp_dot_tempered * mask_negatives, dim=1)))) * mask_positives,dim=1) / positives_per_samples
        
        tcl_loss_mean = torch.mean(tcl_loss)
        return tcl_loss_mean


class SupConWithHardNegatives(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, hard_negative):
        # anchor, positive, hard_negative: [B, D]
        device = anchor.device
        B = anchor.shape[0]

        # Compute similarities
        sim_ap = torch.sum(anchor * positive, dim=-1) / self.temperature  # [B]
        sim_ah = torch.sum(anchor * hard_negative, dim=-1) / self.temperature  # [B]

        # Construct logits: [B, 2] -> (positive, hard negative)
        logits = torch.stack([sim_ap, sim_ah], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=device)  # positive is index 0

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss



class HardNegInfoNCE(nn.Module):
    """
    Anchor-vs-hard-negatives InfoNCE.
    If `positive` is None, uses the anchor itself (self-positive).
    Negatives can be [B, D] or [B, K, D]. L2 norm + temperature softmax.
    Optionally add a small neg_margin to make it harder.
    """
    def __init__(self, temperature: float = 0.07, neg_margin: float = 0.0, detach_neg: bool = True):
        super().__init__()
        self.tau = temperature
        self.neg_m = neg_margin
        self.detach_neg = detach_neg
        self.ce = nn.CrossEntropyLoss()

    def forward(self, anchor: torch.Tensor,
                negatives: torch.Tensor,      # [B, D] or [B, K, D]
                positive: torch.Tensor = None # optional [B, D]
                ) -> torch.Tensor:
        # normalize
        # a = F.normalize(anchor, dim=-1)
        # p = a if positive is None else F.normalize(positive, dim=-1)
        a = anchor
        p = a if positive is None else positive

        n = negatives
        if n.dim() == 2:        # [B, D] -> [B, 1, D]
            n = n.unsqueeze(1)
        # n = F.normalize(n, dim=-1)
        if self.detach_neg:
            n = n.detach()

        # sims
        sim_ap = (a * p).sum(dim=-1, keepdim=True)              # [B, 1]
        sim_an = torch.einsum('bd,bkd->bk', a, n)               # [B, K]
        sim_an = sim_an + self.neg_m                            # small margin on negatives

        # logits / labels
        logits = torch.cat([sim_ap, sim_an], dim=1) / self.tau  # [B, 1+K]
        labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
        return self.ce(logits, labels)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        # x: [B, D], weight: [C, D]
        x = F.normalize(x)
        W = F.normalize(self.weight)
        cos = F.linear(x, W)                        # [B, C]
        sin = torch.sqrt(1.0 - cos**2 + 1e-7)
        phi = cos * self.cos_m - sin * self.sin_m   # cos(theta+m)

        if hasattr(labels, "dtype") and labels.dtype == torch.long:
            one_hot = torch.zeros_like(cos)
            one_hot.scatter_(1, labels.view(-1,1), 1.0)
        else:
            raise ValueError("labels must be LongTensor class ids")

        # use phi on true class, cos elsewhere
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos)
        return logits * self.s

# ---------- New loss: Margin-based hard negatives ----------
class MarginHardNegativesLoss(nn.Module):
    """
    Enforces s_pos >= s_neg + margin on cosine similarity (embeddings assumed L2-normalized).
    Inputs can be [B, D] or broadcastable to a common shape.
    """
    def __init__(self, margin: float = 0.15, reduction: str = "mean", clamp_min: float = 0.0):
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction
        self.clamp_min = clamp_min

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        # cosine similarity (since inputs are L2-normalized, dot == cosine)
        s_pos = (anchor * pos).sum(dim=-1)        # [B] or broadcasted
        s_neg = (anchor * neg).sum(dim=-1)        # [B]

        # margin hinge: max(0, margin - s_pos + s_neg)
        loss = F.relu(self.margin - s_pos + s_neg)

        if self.clamp_min > 0:
            loss = torch.clamp(loss, min=self.clamp_min)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss