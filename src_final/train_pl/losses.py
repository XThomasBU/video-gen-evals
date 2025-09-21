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


def action_consistency_loss(z, y, ratio=True, eps=1e-6):
    S = z @ z.t()
    D = 1.0 - S
    B = D.shape[0]
    eye = torch.eye(B, device=D.device, dtype=torch.bool)
    same = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    diff = (~same) & ~eye

    intra_vals = D[same]
    inter_vals = D[diff]

    if intra_vals.numel() == 0 or inter_vals.numel() == 0:
        return z.new_tensor(0.0, requires_grad=True)

    intra_mean = intra_vals.mean()
    inter_mean = inter_vals.mean()

    if ratio:
        ratio_val = inter_mean / (inter_mean + intra_mean + eps)
        return 1.0 - ratio_val      # non-negative, in [0,1]
    else:
        return intra_mean - inter_mean