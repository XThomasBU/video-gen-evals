import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from utils import *

# set seed
torch.manual_seed(1)
np.random.seed(1)

# ——— CONFIG ———
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 200
WINDOW_SIZE = 32 # 64
STRIDE = 8 # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES"

INPUT_DIM= 1370


def collate_fn(batch):
    sequences, labels, vid_ids, window_ids = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)  # [B, T_max, D]
    labels = torch.tensor(labels)
    return sequences, lengths, labels, vid_ids, window_ids

# class TemporalTransformer(nn.Module):
#     def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, d_model),
#         )
#         self.positional = SinusoidalPositionalEmbedding(d_model)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=4 * d_model,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
#         self.proj = nn.Linear(d_model, latent_dim)
#         self.frame_proj = nn.Linear(d_model, latent_dim)

#     def forward(self, x, lengths=None):  # lengths unused
#         """
#         Args:
#             x: [B, T, input_dim]
#         Returns:
#             x_out: [B, latent_dim]          # sequence-level embedding
#             frame_embeddings: [B, T+1, d_model]   # frame-level (incl. CLS)
#         """
#         x = self.input_proj(x)  # [B, T, d_model]
#         B, T, D = x.shape

#         # Prepend CLS token
#         cls_tokens = self.cls_token.expand(B, 1, D)  # [B, 1, D]
#         x = torch.cat([cls_tokens, x], dim=1)        # [B, T+1, D]

#         x = self.positional(x)                       # [B, T+1, D]
#         x = self.transformer(x)                      # [B, T+1, D]
#         frame_embeddings = x

#         # CLS token output at position 0
#         cls_emb = x[:, 0, :]                         # [B, D]
#         x_out = self.proj(cls_emb)                   # [B, latent_dim]
#         x_out = nn.functional.normalize(x_out, p=2, dim=-1)

#         frame_embeddings = self.frame_proj(frame_embeddings)  # [B, T+1, latent_dim]
#         frame_embeddings = nn.functional.normalize(frame_embeddings, p=2, dim=-1)  # Normalize frame embeddings
#         return x_out, frame_embeddings

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.positional = SinusoidalPositionalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.pre_head_ln = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        self.proj = nn.Linear(2 * d_model, latent_dim)
        self.frame_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x, lengths=None):  # lengths unused
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            x_out: [B, latent_dim]          # sequence-level embedding
            frame_embeddings: [B, T+1, d_model]   # frame-level (incl. CLS)
        """
        x = self.input_proj(x)  # [B, T, d_model]
        B, T, D = x.shape

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, 1, D)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)        # [B, T+1, D]

        x = self.positional(x)                       # [B, T+1, D]
        x = self.transformer(x)                      # [B, T+1, D]
        x = self.pre_head_ln(x)                      # [B, T+1, D]
        frame_embeddings = x

        # CLS token output at position 0
        cls_emb = x[:, 0, :]                 # [B, D]
        mean_emb = x[:, 1:, :].mean(dim=1)   # [B, D]
        concat_emb = torch.cat([cls_emb, mean_emb], dim=-1)  # [B, 2*D]

        x_out = self.proj(concat_emb)        # proj: [2*D → latent_dim]
        x_out = nn.functional.normalize(x_out, p=2, dim=-1)

        frame_embeddings = self.frame_proj(frame_embeddings)  # [B, T+1, latent_dim]
        frame_embeddings = nn.functional.normalize(frame_embeddings, p=2, dim=-1)  # Normalize frame embeddings
        return x_out, frame_embeddings














# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ---- positional embedding (your existing one) ----
# class SinusoidalPositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=10000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div_term)
#         pe[:, 1::2] = torch.cos(pos * div_term)
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: [B, T, D]
#         T = x.size(1)
#         return x + self.pe[:, :T]


# # ---- split projector that only "compresses" ViT, keeps others raw-ish, and applies the SAME projector to ViT motion ----
# class SplitInputProjector(nn.Module):
#     """
#     Expects x: [B, T, 2740] = [vit(1024), others(346), motion_full(1370)]
#     where motion_full = [vit_motion(1024), others_motion(346)]

#     Projects ViT (static + motion) with a shared linear (1024 -> d_vit_proj).
#     Others (static + motion) get LayerNorm only (kept near-raw).
#     Concats -> GELU -> Linear to d_out (Transformer d_model).
#     """
#     def __init__(self, d_vit=1024, d_others=346, d_vit_proj=128, d_out=256):
#         super().__init__()
#         self.d_vit = d_vit
#         self.d_others = d_others
#         self.d_motion = d_vit + d_others  # 1370

#         # 1) ViT projection (shared by static and motion parts)
#         self.vit_proj = nn.Linear(d_vit, d_vit_proj)

#         # 2) Light norm for others static/motion
#         self.others_ln = nn.LayerNorm(d_others)
#         self.others_motion_ln = nn.LayerNorm(d_others)

#         # 3) Mixer to Transformer dim
#         # concat = vit_proj + others + vit_motion_proj + others_motion  -> d_vit_proj + d_others + d_vit_proj + d_others
#         d_cat = (2 * d_vit_proj) + (2 * d_others)
#         self.mixer = nn.Linear(d_cat, d_out)
#         self.act = nn.GELU()

#     def forward(self, x):
#         """
#         x: [B, T, 2740] = [vit, others, motion_full]
#         returns: [B, T, d_out]
#         """
#         vit = x[..., :self.d_vit]  # [B,T,1024]
#         others = x[..., self.d_vit:self.d_vit + self.d_others]  # [B,T,346]
#         motion_full = x[..., self.d_vit + self.d_others:]  # [B,T,1370]

#         vit_m = motion_full[..., :self.d_vit]                 # [B,T,1024]
#         others_m = motion_full[..., self.d_vit:]              # [B,T,346]

#         # project ViT static and motion with SAME weights
#         vit_p = self.vit_proj(vit)                            # [B,T,d_vit_proj]
#         vit_m_p = self.vit_proj(vit_m)                        # [B,T,d_vit_proj]

#         # keep others almost raw, just normalize
#         others_n = self.others_ln(others)                     # [B,T,346]
#         others_m_n = self.others_motion_ln(others_m)          # [B,T,346]

#         cat = torch.cat([vit_p, others_n, vit_m_p, others_m_n], dim=-1)  # [B,T, 2*d_vit_proj + 2*d_others]
#         # out = self.mixer(self.act(cat))                       # [B,T,d_out]
#         return cat


# class TemporalTransformer(nn.Module):
#     def __init__(self,
#                  latent_dim,
#                  d_model=256,
#                  n_heads=4,
#                  n_layers=2,
#                  dropout=0.1,
#                  d_vit=1024,
#                  d_others=346,
#                  d_vit_proj=128):
#         """
#         Input per frame is assumed: [vit(1024), others(346), motion_full(1370)]
#         motion_full = diff of the same 1370-d vector -> split as [vit_motion(1024), others_motion(346)]
#         """
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

#         # Only heavily project ViT (static + motion). Others kept raw-ish.
#         self.input_proj = SplitInputProjector(
#             d_vit=d_vit,
#             d_others=d_others,
#             d_vit_proj=d_vit_proj,
#             d_out=d_model
#         )

#         self.positional = SinusoidalPositionalEmbedding(d_model)

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=4 * d_model,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu'
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
#         self.pre_head_ln = nn.LayerNorm(d_model)

#         # Pool with [CLS || mean] then project to embedding space
#         self.proj = nn.Linear(d_model, latent_dim)
#         self.frame_proj = nn.Linear(d_model, latent_dim)

#     def forward(self, x, lengths=None):
#         """
#         x: [B, T, 2740] = [vit(1024), others(346), motion_full(1370)]
#         Returns:
#           x_out: [B, latent_dim]                 # sequence-level embedding (L2-normalized)
#           frame_embeddings: [B, T+1, latent_dim]# token-level (CLS + frames), L2-normalized
#         """
#         # 1) group-aware projection to Transformer width
#         x = self.input_proj(x)                     # [B,T,d_model]
#         B, T, D = x.shape

#         # 2) prepend CLS
#         cls = self.cls_token.expand(B, 1, D)       # [B,1,D]
#         x = torch.cat([cls, x], dim=1)             # [B,T+1,D]

#         # 3) pos + transformer
#         x = self.positional(x)
#         x = self.transformer(x)
#         x = self.pre_head_ln(x)

#         # 4) pooling: concat(CLS, mean)
#         cls_emb = x[:, 0, :]                       # [B,D]
#         mean_emb = x[:, 1:, :].mean(dim=1)         # [B,D]
#         # concat_emb = torch.cat([cls_emb, mean_emb], dim=-1)  # [B,2D]

#         x_out = self.proj(cls_emb)              # [B,latent_dim]
#         x_out = F.normalize(x_out, p=2, dim=-1)

#         frame_embeddings = self.frame_proj(x)      # [B,T+1,latent_dim]
#         frame_embeddings = F.normalize(frame_embeddings, p=2, dim=-1)

#         return x_out, frame_embeddings















# import torch
# from torch import nn

# class TemporalTransformerWithCrossAttnPool(nn.Module):
#     def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, d_model),
#         )
#         self.positional = SinusoidalPositionalEmbedding(d_model)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=4 * d_model,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

#         # Learnable pooling query (replaces CLS)
#         self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
#         self.pool_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

#         self.proj = nn.Linear(d_model, latent_dim)
#         self.frame_proj = nn.Linear(d_model, latent_dim)

#     def forward(self, x, lengths=None):  # lengths unused
#         """
#         Args:
#             x: [B, T, input_dim]
#         Returns:
#             x_out: [B, latent_dim]          # sequence-level embedding
#             frame_embeddings: [B, T, d_model]   # frame-level outputs (no CLS)
#         """
#         x = self.input_proj(x)  # [B, T, d_model]
#         B, T, D = x.shape

#         # Add positional encoding (no CLS prepended)
#         x = self.positional(x)  # [B, T, D]
#         x = self.transformer(x) # [B, T, D]
#         frame_embeddings = x    # [B, T, D]

#         # Cross-attention pooling: each pool_query attends to all frames in each window
#         pool_query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
#         # pool_attn expects [B, L, D]: query=[B, 1, D], key/value=[B, T, D]
#         pooled, _ = self.pool_attn(pool_query, frame_embeddings, frame_embeddings)  # [B, 1, D]
#         pooled = pooled.squeeze(1)  # [B, D]

#         x_out = self.proj(pooled)  # [B, latent_dim]
#         x_out = nn.functional.normalize(x_out, p=2, dim=-1)

#         frame_embeddings = self.frame_proj(frame_embeddings)  # [B, T, latent_dim]
#         frame_embeddings = nn.functional.normalize(frame_embeddings, p=2, dim=-1)

#         return x_out, frame_embeddings


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


# ——— TRAINING ———
def train():
    print(" Loading datasets...")

    if os.path.exists(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt"):
        train_samples = torch.load(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_labels = torch.load(f"SAVE_NEW/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_samples = torch.load(f"SAVE_NEW/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_labels = torch.load(f"SAVE_NEW/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_vid_ids = torch.load(f"SAVE_NEW/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_vid_ids = torch.load(f"SAVE_NEW/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_window_ids = torch.load(f"SAVE_NEW/train_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_window_ids = torch.load(f"SAVE_NEW/test_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

        # # combine all tets values to train
        # train_samples = train_samples + test_samples
        # train_labels = train_labels + test_labels
        # train_vid_ids = train_vid_ids + test_vid_ids
        # train_window_ids = train_window_ids + test_window_ids

        
        class PoseVideoDatasetFromTensors(Dataset):
            def __init__(self, samples, labels, vid_ids, window_ids):
                self.samples = torch.stack(samples)
                self.labels = torch.tensor(labels)
                self.vid_ids = vid_ids
                self.window_ids = window_ids

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx], self.labels[idx], self.vid_ids[idx], self.window_ids[idx]


        train_dataset = PoseVideoDatasetFromTensors(train_samples, train_labels, train_vid_ids, train_window_ids)
        test_dataset = PoseVideoDatasetFromTensors(test_samples, test_labels, test_vid_ids, test_window_ids)
        print(f" Loaded existing datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")
        print(f" Train samples: {train_dataset.samples.shape}, Train labels: {train_dataset.labels.shape}, Train vid_ids: {len(train_dataset.vid_ids)}, Train window_ids: {len(train_dataset.window_ids)}")
        print(f" Test samples: {test_dataset.samples.shape}, Test labels: {test_dataset.labels.shape}, Test vid_ids: {len(test_dataset.vid_ids)}, Test window_ids: {len(test_dataset.window_ids)}")
        print(" Using existing datasets for training...")
    
    # sampler = EnsurePositivesSampler(train_labels, batch_size=BATCH_SIZE, min_pos_per_class=4)
    sampler = PKBatchSampler(train_labels, P=10, K=26)
    train_loader = DataLoader(
        train_dataset,
        # batch_size=BATCH_SIZE,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(" Building model...")
    input_dim = train_dataset[0][0].shape[-1]
    model = TemporalTransformer(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)

    # print number of parameters
    print(f" Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # loss_fn = TCL()
    loss_fn = SUPCON()
    loss_hard = SupConWithHardNegatives()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_iter = len(train_loader) * EPOCHS
    print(f" Total iterations: {num_iter}")
    cosine_annel = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-6)

    all_window_ids = [w for _, _, _, w in train_dataset]
    all_vid_ids = [v for _, _, v, _ in train_dataset]

    loss_dict = {'loss': [], 'loss_con': [], 'loss_hard': [], 'smoothness': [], 'hard_shuffle_big': [], 'hard_shuffle_small': [], 'reverse': [], 'static': []}

    print(" Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_loss_con = 0
        total_loss_hard = 0
        total_smoothness_loss = 0
        total_hard_shuffle_big_loss = 0
        total_hard_shuffle_small_loss = 0
        total_reverse_loss = 0
        total_static_loss = 0
        # if epoch == 0:
        #     print(f"🔎 Plotting t-SNE (joint) for epoch 0")
        #     model.eval()
        #     # Get all train
        #     all_train_embeds, all_train_labels = [], []
        #     with torch.no_grad():
        #         for seqs, lengths, labels, vid_ids in train_loader:
        #             seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
        #             emb, _ = model(seqs, lengths)
        #             all_train_embeds.append(emb.cpu())
        #             all_train_labels.append(labels.cpu())
        #     all_train_embeds = torch.cat(all_train_embeds)
        #     all_train_labels = torch.cat(all_train_labels)

        #     # Get all test
        #     all_test_embeds, all_test_labels = [], []
        #     with torch.no_grad():
        #         for seqs, lengths, labels, vid_ids in test_loader:
        #             seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
        #             emb, _ = model(seqs, lengths)
        #             all_test_embeds.append(emb.cpu())
        #             all_test_labels.append(labels.cpu())
        #     all_test_embeds = torch.cat(all_test_embeds)
        #     all_test_labels = torch.cat(all_test_labels)

        #     # Plot
        #     plot_embeddings_epoch(
        #         -1,
        #         all_train_embeds, all_train_labels,
        #         all_test_embeds, all_test_labels,
        #         ALL_CLASSES,
        #         save_dir="SAVE_NEW2/tsne_joint"
        #     )
            # model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            seqs, lengths, labels, vid_ids, window_ids = batch
            seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            shuffled_seqs = partial_shuffle_within_window(seqs, lengths, vid_ids)


            optimizer.zero_grad()
            embeddings, frame_embeddings = model(seqs, lengths)

            # current_window_embeddings = []
            # next_window_embeddings = []

            # for i in range(len(vid_ids)):
            #     next_idx = get_next_window_index(all_vid_ids, all_window_ids, vid_ids[i], window_ids[i])
            #     if next_idx is not None:
            #         # Get the next window's input (you might want to cache/preload these for speed!)
            #         next_seq, next_length, _, _ = train_dataset[next_idx]
            #         next_seq = next_seq.unsqueeze(0).to(DEVICE)
            #         next_length = torch.tensor([next_length]).to(DEVICE)
            #         with torch.no_grad():
            #             next_emb, _ = model(next_seq, next_length)
            #         current_window_embeddings.append(embeddings[i])         # embeddings for the current batch
            #         next_window_embeddings.append(next_emb.squeeze(0))      # next window's embedding
            # if current_window_embeddings:  # Only compute if non-empty
            #     current_window_embeddings = torch.stack(current_window_embeddings)  # [N, D]
            #     next_window_embeddings = torch.stack(next_window_embeddings)  # [N, D]
            #     l2_smoothness_loss = (current_window_embeddings - next_window_embeddings).norm(dim=1).mean()

            shuffled_embeddings, _ = model(shuffled_seqs, lengths)

            reverse_seqs = reverse_sequence(seqs, lengths)
            reverse_embeddings, _ = model(reverse_seqs, lengths)

            # get cosine sim 
            cosine_sim = torch.cosine_similarity(embeddings.unsqueeze(1), shuffled_embeddings.unsqueeze(0), dim=-1)  # [B, B]         
            loss_org = loss_fn(embeddings, labels)

            shuffled_seqs_small = partial_shuffle_within_window(seqs, lengths, vid_ids, shuffle_fraction=0.3)
            shuffled_embeddings_small, _ = model(shuffled_seqs_small, lengths)

            # # # ——— temporal smoothness penalty ———
            # # # build a map: vid_id → list of batch‐indices
            # vid_to_indices = defaultdict(list)
            # for i, vid in enumerate(vid_ids):
            #     vid_to_indices[vid].append(i)

            static_seqs = get_static_window(seqs)
            static_embeddings, _ = model(static_seqs, lengths)

            shuffled_big_loss = loss_hard(embeddings, embeddings, shuffled_embeddings)
            shuffled_small_loss = loss_hard(embeddings, embeddings, shuffled_embeddings_small)
            reverse_loss = loss_hard(embeddings, embeddings, reverse_embeddings)
            static_loss = loss_hard(embeddings, embeddings, static_embeddings)

            # Combine hard losses
            loss_hard_combined = shuffled_big_loss + reverse_loss + static_loss
            # print(cosine_sim.mean().item(), loss_org.item(), loss_hard_combined.item())

            # frame smoothness loss
            frame_smoothness_loss = second_order_steady_loss(frame_embeddings[:, 1:])  # exclude CLS token
            # frame_smoothness_loss = second_order_steady_loss(frame_embeddings)  # exclude CLS token
            # print(l2_smoothness_loss)

            loss = loss_org +  10 * loss_hard_combined 
            # loss = loss_org
            # loss = loss_org + 10 * loss_hard_combined
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_con += loss_org.item()
            total_loss_hard += loss_hard_combined.item()
            total_smoothness_loss += frame_smoothness_loss.item()
            total_hard_shuffle_big_loss += shuffled_big_loss.item()
            total_hard_shuffle_small_loss += shuffled_small_loss.item()
            total_reverse_loss += reverse_loss.item()
            total_static_loss += static_loss.item()
            cosine_annel.step()
        print(f" Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        loss_dict['loss'].append(total_loss / len(train_loader))
        loss_dict['loss_con'].append(total_loss_con / len(train_loader))
        loss_dict['loss_hard'].append(total_loss_hard / len(train_loader))
        loss_dict['smoothness'].append(total_smoothness_loss / len(train_loader))
        loss_dict['hard_shuffle_big'].append(total_hard_shuffle_big_loss / len(train_loader))
        loss_dict['hard_shuffle_small'].append(total_hard_shuffle_small_loss / len(train_loader))
        loss_dict['reverse'].append(total_reverse_loss / len(train_loader))
        loss_dict['static'].append(total_static_loss / len(train_loader))
        
        # if (epoch) % 20 == 0 or epoch == 0:
        #     print(f"🔎 Plotting t-SNE (joint) for epoch {epoch+1}")
        #     model.eval()
        #     # Get all train
        #     all_train_embeds, all_train_labels = [], []
        #     with torch.no_grad():
        #         for seqs, lengths, labels, vid_ids in train_loader:
        #             seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
        #             emb, _ = model(seqs, lengths)
        #             all_train_embeds.append(emb.cpu())
        #             all_train_labels.append(labels.cpu())
        #     all_train_embeds = torch.cat(all_train_embeds)
        #     all_train_labels = torch.cat(all_train_labels)

        #     # Get all test
        #     all_test_embeds, all_test_labels = [], []
        #     with torch.no_grad():
        #         for seqs, lengths, labels, vid_ids in test_loader:
        #             seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
        #             emb, _ = model(seqs, lengths)
        #             all_test_embeds.append(emb.cpu())
        #             all_test_labels.append(labels.cpu())
        #     all_test_embeds = torch.cat(all_test_embeds)
        #     all_test_labels = torch.cat(all_test_labels)

        #     # Plot
        #     plot_embeddings_epoch(
        #         epoch,
        #         all_train_embeds, all_train_labels,
        #         all_test_embeds, all_test_labels,
        #         ALL_CLASSES,
        #         save_dir="SAVE_NEW2/tsne_joint"
        #     )
        #     model.train()

    print(" Saving model...")
    torch.save(model.state_dict(), f"SAVE_NEW2/temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    # model.load_state_dict(torch.load(f"SAVE_NEW2/temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt", map_location=DEVICE))

    # plot all loss curves, each loss as a subplot
    fig, axs = plt.subplots(8, figsize=(20, 24))

    axs[0].plot(loss_dict['loss'], label='Total Loss', color='royalblue', linewidth=2.5)
    axs[0].set_title('Total Loss')
    axs[0].legend()

    axs[1].plot(loss_dict['loss_con'], label='Contrastive Loss', color='darkorange', linewidth=2.5)
    axs[1].set_title('Contrastive Loss')
    axs[1].legend()

    axs[2].plot(loss_dict['loss_hard'], label='Hard Loss', color='seagreen', linewidth=2.5)
    axs[2].set_title('Hard Loss')
    axs[2].legend()

    axs[3].plot(loss_dict['smoothness'], label='Smoothness Loss', color='crimson', linewidth=2.5)
    axs[3].set_title('Smoothness Loss')
    axs[3].legend()

    axs[4].plot(loss_dict['hard_shuffle_big'], label='Hard Shuffle Big Loss', color='purple', linewidth=2.5)
    axs[4].set_title('Hard Shuffle Big Loss')
    axs[4].legend()

    axs[5].plot(loss_dict['hard_shuffle_small'], label='Hard Shuffle Small Loss', color='teal', linewidth=2.5)
    axs[5].set_title('Hard Shuffle Small Loss')
    axs[5].legend()

    axs[6].plot(loss_dict['reverse'], label='Reverse Loss', color='maroon', linewidth=2.5)
    axs[6].set_title('Reverse Loss')
    axs[6].legend()

    axs[7].plot(loss_dict['static'], label='Static Window Loss', color='goldenrod', linewidth=2.5)
    axs[7].set_title('Static Window Loss')
    axs[7].legend()

    plt.tight_layout()
    plt.savefig(f"SAVE_NEW2/loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")
    print(f" Saved as loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    print(" Computing train embeddings...")
    model.eval()
    all_train_embeds = []
    all_train_labels = []
    all_train_vid_ids = []   # <-- NEW
    with torch.no_grad():
        for seqs, lengths, labels, vid_ids, window_ids in tqdm(train_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(seqs, lengths)
            all_train_embeds.append(emb.cpu())
            all_train_labels.append(labels.cpu())
            all_train_vid_ids.extend(vid_ids)  # vid_ids should be a list/tuple of strings or ints
    all_train_embeds = torch.cat(all_train_embeds)
    all_train_labels = torch.cat(all_train_labels)
    all_train_vid_ids = np.array(all_train_vid_ids)

    # save all_train_embeds and all_train_labels
    torch.save(all_train_embeds, f"SAVE_NEW2/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(all_train_labels, f"SAVE_NEW2/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(all_train_vid_ids, f"SAVE_NEW2/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    # Compute class centroids
    centroids = {}
    for cls in torch.unique(all_train_labels):
        mask_cls = (all_train_labels == cls)
        embeds_cls = all_train_embeds[mask_cls]
        vid_ids_cls = all_train_vid_ids[mask_cls.numpy()]
        
        unique_vids = np.unique(vid_ids_cls)
        video_embeds = []
        for vid in unique_vids:
            vid_mask = (vid_ids_cls == vid)
            video_mean = embeds_cls[vid_mask].mean(dim=0)
            video_embeds.append(video_mean)
        centroid = torch.stack(video_embeds, dim=0).mean(dim=0)
        centroids[int(cls.item())] = centroid

    print(" Evaluating on test set...")
    model.eval()

    # Collect distances per (true_class, other_class)
    per_class_same_dists = {cls: [] for cls in centroids}
    per_class_to_other_dists = {cls: {other: [] for other in centroids if other != cls} for cls in centroids}

    with torch.no_grad():
        for seqs, lengths, labels, vid_ids, window_ids in tqdm(test_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(seqs, lengths)
            emb = emb.cpu()
            for e, l in zip(emb, labels):
                l = int(l.item())

                # Distance to own centroid
                dist_same = torch.norm(e - centroids[l]).item()
                per_class_same_dists[l].append(dist_same)

                # Distance to each other class centroid
                for c in centroids:
                    if c != l:
                        dist_other = torch.norm(e - centroids[c]).item()
                        per_class_to_other_dists[l][c].append(dist_other)

    # ---------- Pretty print results ----------
    print("\n Per-Class Distance Statistics:")

    for cls in centroids:
        same_dists = per_class_same_dists[cls]
        if same_dists:
            print(f"\nClass {cls} ({ALL_CLASSES[cls]}):")
            print(f"  Mean INTRA-class distance (own centroid): {np.mean(same_dists):.4f} ± {np.std(same_dists):.4f}")

            print("  INTER-class distances to other centroids:")
            for other_cls in centroids:
                if other_cls == cls:
                    continue
                other_dists = per_class_to_other_dists[cls][other_cls]
                if other_dists:
                    print(f"    -> to Class {other_cls} ({ALL_CLASSES[other_cls]}): "
                        f"{np.mean(other_dists):.4f} ± {np.std(other_dists):.4f}")

    print("\n Consistency Scores:")

    consistency_scores = {}
    for cls in centroids:
        intra_mean = np.mean(per_class_same_dists[cls]) if per_class_same_dists[cls] else np.nan

        inter_dists_all = []
        for other_cls in centroids:
            if other_cls == cls:
                continue
            inter_dists_all.extend(per_class_to_other_dists[cls][other_cls])

        inter_mean = np.mean(inter_dists_all) if inter_dists_all else np.nan

        if np.isnan(intra_mean) or np.isnan(inter_mean) or (intra_mean + inter_mean) == 0:
            score = float('nan')
        else:
            score = inter_mean / (inter_mean + intra_mean)

        consistency_scores[cls] = score
        print(f"  Class {cls} ({ALL_CLASSES[cls]}): Consistency Score = {score:.4f}")

    print("\n Overall Class Consistency Scores:")
    pprint({ALL_CLASSES[k]: v for k, v in consistency_scores.items()})

    with open(f"SAVE_NEW2/centroids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pkl", "wb") as f:
        pickle.dump(centroids, f)

    print("\n Done!")

    print("\n Visualizing window embeddings and centroids...")

    # Compute test embeddings
    print(" Computing test embeddings...")
    all_test_embeds = []
    all_test_labels = []
    with torch.no_grad():
        for seqs, lengths, labels, vid_ids, window_ids in tqdm(test_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(seqs, lengths)
            emb = emb.cpu()
            all_test_embeds.append(emb)
            all_test_labels.append(labels)
    all_test_embeds = torch.cat(all_test_embeds)
    all_test_labels = torch.cat(all_test_labels)

    # ---------- Combine for joint PCA ----------
    # combined_embeds = torch.cat([all_train_embeds, all_test_embeds], dim=0).numpy()
    # pca = TSNE(n_components=2)
    # projected_all = pca.fit_transform(combined_embeds)

    # projected_train_embeds = projected_all[:len(all_train_embeds)]
    # projected_test_embeds = projected_all[len(all_train_embeds):]
    # projected_centroids = pca.transform(torch.stack([centroids[c] for c in sorted(centroids)]).numpy())
    combined_embeds = torch.cat([all_train_embeds, all_test_embeds], dim=0).numpy()
    centroid_matrix = torch.stack([centroids[c] for c in sorted(centroids)]).numpy()
    full_matrix = np.concatenate([combined_embeds, centroid_matrix], axis=0)

    pca = TSNE(n_components=2)
    projected_all = pca.fit_transform(full_matrix)

    # Then split back:
    projected_train_embeds = projected_all[:len(all_train_embeds)]
    projected_test_embeds = projected_all[len(all_train_embeds):len(all_train_embeds) + len(all_test_embeds)]
    projected_centroids = projected_all[-len(centroids):]

    # ---------- Plot ----------
    colors = plt.cm.get_cmap("tab10", len(ALL_CLASSES))

    plt.figure(figsize=(10, 8))

    # Plot train embeddings
    for cls in range(len(ALL_CLASSES)):
        mask = (all_train_labels == cls).numpy()
        plt.scatter(
            projected_train_embeds[mask, 0],
            projected_train_embeds[mask, 1],
            s=10,
            color=colors(cls),
            label=ALL_CLASSES[cls],
            alpha=0.6
        )

    # Plot centroids
    for i, (x, y) in enumerate(projected_centroids):
        plt.scatter(x, y, color=colors(i), edgecolors='k', s=200, marker='X', linewidths=2)
        plt.text(x, y + 0.05, ALL_CLASSES[i], fontsize=9, ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.8))

    # Plot test embeddings with outline using star shape
    for cls in range(len(ALL_CLASSES)):
        mask = (all_test_labels == cls).numpy()
        plt.scatter(
            projected_test_embeds[mask, 0],
            projected_test_embeds[mask, 1],
            s=100,
            facecolors=colors(cls),
            edgecolors='black',
            linewidths=0.5,
            marker='*',  # Changed marker to star
            label=f"{ALL_CLASSES[cls]} (test)",
            alpha=0.9
        )

    plt.title("2D Projection of Train + Test Window Embeddings + Centroids")
    # plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"SAVE_NEW2/embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png", dpi=200)
    print(f" Saved as embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    results = test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE)

# ——— MAIN ———
if __name__ == "__main__":
    train()