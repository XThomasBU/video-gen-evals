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
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

# set seed
torch.manual_seed(1)
np.random.seed(1)

# ——— CONFIG ———
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 100
WINDOW_SIZE = 32 # 64
STRIDE = 8 # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES"

INPUT_DIM= 1370

def partial_shuffle_within_window(seqs, lengths, vid_ids, shuffle_fraction=0.7):
    shuffled = seqs.clone()
    batch_size, max_len, feat_dim = seqs.shape
    for i in range(batch_size):
        l = lengths[i]
        if l > 1:
            n_to_shuffle = max(1, int(shuffle_fraction * l))
            indices = torch.randperm(l)[:n_to_shuffle]
            shuffled_part = shuffled[i, indices][torch.randperm(n_to_shuffle)]
            shuffled[i, indices] = shuffled_part
    return shuffled

def reverse_sequence(seqs, lengths):
    # [B, T, D] → reversed in T dim
    reversed_seqs = []
    for i, l in enumerate(lengths):
        reversed = torch.flip(seqs[i, :l], dims=[0])
        pad_len = seqs.shape[1] - l
        if pad_len > 0:
            pad = torch.zeros(pad_len, seqs.shape[2], device=seqs.device)
            reversed = torch.cat([reversed, pad], dim=0)
        reversed_seqs.append(reversed)
    return torch.stack(reversed_seqs, dim=0)

# ——— LOAD VIDEO FRAMES ———
def load_video_sequence(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh", POSE_DIR)
    twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

    for idx, p in enumerate(frames):
        # try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        params = data["pred_smpl_params"]

        if isinstance(params, list):
            if len(params) < 1:
                continue
            params = params[0]
        if not isinstance(params, dict):
            continue

        vit_feature   = np.array(params["token_out"]).flatten()
        global_orient = np.array(params["global_orient"]).flatten()
        body_pose     = np.array(params["body_pose"]).flatten()
        betas         = np.array(params["betas"]).flatten()

        twod_point_path = twod_points_paths[idx]
        twod_kp = np.load(twod_point_path).flatten()
        twod_kp = twod_kp[:120]  # Ensure 120 keypoints

        # # Normalize each part
        vit_feature   /= np.linalg.norm(vit_feature) + 1e-8
        global_orient /= np.linalg.norm(global_orient) + 1e-8
        body_pose     /= np.linalg.norm(body_pose) + 1e-8
        betas         /= np.linalg.norm(betas) + 1e-8
        twod_kp       /= np.linalg.norm(twod_kp) + 1e-8

        # if INPUT_DIM == 1250:
        #     vec = np.concatenate([vit_feature, global_orient, body_pose, betas], axis=0)
        #     vec = vec / np.linalg.norm(vec) + 1e-8
        #     if vec.shape[0] != 1250:
        #         continue
        # elif INPUT_DIM == 226:
        #     vec = np.concatenate([global_orient, body_pose, betas], axis=0)
        #     vec = vec / np.linalg.norm(vec) + 1e-8
        #     if vec.shape[0] != 226:
        #         continue
        # elif INPUT_DIM == 1250 + 36:
        vec = np.concatenate([vit_feature, global_orient, body_pose, betas, twod_kp], axis=0)
        # vec = vec / np.linalg.norm(vec) + 1e-8
        # if vec.shape[0] != 1250 + 36:
        #     continue

        frame_vecs.append(torch.tensor(vec, dtype=torch.float32))
        # except:
        #     continue

    if len(frame_vecs) < 2:
        return None

    # [T, 1250]
    frame_tensor = torch.stack(frame_vecs, dim=0)

    # Compute motion vectors (frame-to-frame deltas)
    motion_vecs = frame_tensor[1:] - frame_tensor[:-1]  # [T-1, 1250]
    motion_vecs = torch.cat([torch.zeros(1, INPUT_DIM), motion_vecs], dim=0)  # [T, 1250]


    # Concatenate original + motion
    enriched_tensor = torch.cat([frame_tensor, motion_vecs], dim=1)  # [T, 2500] # 2500 = 1250 * 2
    # print(enriched_tensor.shape)
    # exit()

    # # return enriched_tensor
    # if INPUT_DIM == (1250 + 36) * 2:
    #     return enriched_tensor
    # return frame_tensor
    return enriched_tensor

# ——— SLIDING WINDOW ———
def extract_windows(seq, window_size, stride):
    windows = []
    num_frames = seq.shape[0]
    for start in range(0, num_frames, stride):
        end = start + window_size
        if end > num_frames:
            # pad_len = end - num_frames
            # pad = seq[-1:].repeat(pad_len, 1)
            # window = torch.cat([seq[start:], pad], dim=0)
            continue
        else:
            window = seq[start:end]
        windows.append(window)
        if end >= num_frames:
            break
    return windows

# def extract_windows(seq, min_size=16, max_size=128, stride=8):
#     windows = []
#     num_frames = seq.shape[0]
#     for start in range(0, num_frames, stride):
#         window_size = np.random.randint(min_size, max_size + 1)
#         end = start + window_size
#         if end > num_frames:
#             pad_len = end - num_frames
#             pad = seq[-1:].repeat(pad_len, 1)
#             window = torch.cat([seq[start:], pad], dim=0)
#         else:
#             window = seq[start:end]
#         windows.append(window)
#         if end >= num_frames:
#             break
#     return windows

# ——— DATASET ———
# class PoseVideoDataset(Dataset):
#     def __init__(self, root, classes, window_size=64, stride=32, split="train"):
#         self.samples = []
#         self.labels = []
#         self.class_to_idx = {c: i for i, c in enumerate(classes)}

#         self.per_class_data = {c: [] for c in classes}
#         for cls in classes:
#             class_dir = Path(root) / cls
#             for vid in os.listdir(class_dir):
#                 vid_path = class_dir / vid
#                 seq = load_video_sequence(vid_path)
#                 if seq is None:
#                     continue
#                 windows = extract_windows(seq, window_size, stride)
#                 self.per_class_data[cls].extend(windows)

#         # Train/test split
#         self.samples = []
#         self.labels = []
#         for cls in classes:
#             all_windows = self.per_class_data[cls]
#             np.random.shuffle(all_windows)
#             n_train = int(0.8 * len(all_windows))
#             if split == "train":
#                 selected = all_windows[:n_train]
#             else:
#                 selected = all_windows[n_train:]
#             self.samples.extend(selected)
#             self.labels.extend([self.class_to_idx[cls]] * len(selected))

#         print(f"✅ Loaded {len(self.samples)} {split} windows")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx], self.labels[idx]
class PoseVideoDataset(Dataset):
    def __init__(self, root, classes, window_size=64, stride=32, split="train"):
        self.samples = []
        self.labels = []
        self.vid_ids = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.per_class_video_paths = {c: [] for c in classes}
        for cls in classes:
            class_dir = Path(root) / cls
            videos = [class_dir / vid for vid in os.listdir(class_dir)]
            self.per_class_video_paths[cls] = videos

        # Split videos *per class*
        self.video_split = {}
        for cls in classes:
            vids = self.per_class_video_paths[cls]
            np.random.shuffle(vids)
            n_train = int(0.8 * len(vids))
            if split == "train":
                selected_videos = vids[:n_train]
            else:
                selected_videos = vids[n_train:]
            self.video_split[cls] = selected_videos

        # Now extract all windows from the selected videos
        for cls in tqdm(classes, desc=f"Loading {split} videos", total=len(classes)):
            videos = self.video_split[cls]
            for idx, vid_path in enumerate(tqdm(videos, desc=f"Loading {cls} videos", total=len(videos))):
                seq = load_video_sequence(vid_path)
                if seq is None:
                    continue
                windows = extract_windows(seq, WINDOW_SIZE, STRIDE)
                self.samples.extend(windows)
                self.labels.extend([self.class_to_idx[cls]] * len(windows))
                self.vid_ids.extend([vid_path.name] * len(windows))
                # if idx == 9:
                #     break

        print(f"✅ Loaded {len(self.samples)} {split} windows from {split} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.vid_ids[idx]

# ——— COLLATE FN ———
# def collate_fn(batch):
#     sequences, labels, vid_ids = zip(*batch)
#     sequences = torch.stack(sequences)
#     labels = torch.tensor(labels)
#     lengths = torch.full((len(labels),), sequences.shape[1], dtype=torch.long)
#     return sequences, lengths, labels, vid_ids
def collate_fn(batch):
    sequences, labels, vid_ids = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)  # [B, T_max, D]
    labels = torch.tensor(labels)
    return sequences, lengths, labels, vid_ids

# ——— MODEL ———
# class TemporalTransformer(nn.Module):
#     def __init__(self, input_dim, latent_dim, n_heads=1, n_layers=2, dropout=0.1):
#         super().__init__()
#         self.positional = nn.Parameter(torch.randn(512, input_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim,
#             nhead=n_heads,
#             dim_feedforward=4 * input_dim,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.proj = nn.Linear(input_dim, latent_dim)

#     def forward(self, x, lengths):
#         batch_size, max_len, _ = x.shape
#         pos_emb = self.positional[:max_len, :].unsqueeze(0)
#         x = x + pos_emb

#         mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
#         x = self.transformer(x, src_key_padding_mask=mask)

#         mask_float = (~mask).unsqueeze(-1).float()
#         summed = (x * mask_float).sum(dim=1)
#         counts = mask_float.sum(dim=1).clamp(min=1e-6)
#         x = summed / counts

#         x = self.proj(x)
#         x = nn.functional.normalize(x, p=2, dim=-1)
#         return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]
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

#         self.attn_query = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
#         self.proj = nn.Linear(d_model, latent_dim)

#     def forward(self, x, lengths):
#         x = self.input_proj(x)  # [B, T, d_model]
#         B, T, D = x.shape

#         # pos_emb = self.positional[:T, :].unsqueeze(0)
#         # x = x + pos_emb
#         x = self.positional(x)

#         # Mask: True where padding is applied
#         mask = torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]
#         x = self.transformer(x, src_key_padding_mask=mask)  # [B, T, D]
#         frame_embeddings = x

#         # Attention pooling: Q = [1, 1, D], K,V = [B, T, D] → out = [B, 1, D]
#         q = self.attn_query.expand(B, -1, -1)  # [B, 1, D]
#         attn_weights = torch.softmax((q @ x.transpose(1, 2)) / (D ** 0.5), dim=-1)  # [B, 1, T]
#         pooled = attn_weights @ x  # [B, 1, D]
#         pooled = pooled.squeeze(1)  # [B, D]

#         x = self.proj(pooled)
#         x = nn.functional.normalize(x, p=2, dim=-1)
#         return x, frame_embeddings
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, d_model),
        )
        self.positional = SinusoidalPositionalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        self.proj = nn.Linear(d_model, latent_dim)

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
        frame_embeddings = x

        # CLS token output at position 0
        cls_emb = x[:, 0, :]                         # [B, D]
        x_out = self.proj(cls_emb)                   # [B, latent_dim]
        x_out = nn.functional.normalize(x_out, p=2, dim=-1)
        return x_out, frame_embeddings

# ——— CONTRASTIVE LOSS ———
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

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
class EnsurePositivesSampler(Sampler):
    def __init__(self, labels, batch_size, min_pos_per_class=2, max_classes_per_batch=None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.min_pos_per_class = min_pos_per_class
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        self.classes = list(self.label_to_indices.keys())
        self.max_classes_per_batch = min(
            max_classes_per_batch or (batch_size // min_pos_per_class),
            len(self.classes)
        )

    def __iter__(self):
        idxs = []
        rng = np.random.default_rng()
        while True:
            # Pick random classes for this batch
            chosen_classes = rng.choice(self.classes, size=self.max_classes_per_batch, replace=False)
            batch = []
            for c in chosen_classes:
                inds = self.label_to_indices[c]
                if len(inds) >= self.min_pos_per_class:
                    sampled = rng.choice(inds, size=self.min_pos_per_class, replace=False)
                else:
                    sampled = rng.choice(inds, size=self.min_pos_per_class, replace=True)
                batch.extend(sampled)
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            idxs.extend(batch)
            if len(idxs) >= len(self.labels):
                break
        return iter(idxs)

    def __len__(self):
        return len(self.labels)


def partial_shuffle_within_window(seqs, lengths, vid_ids, shuffle_fraction=0.7):
    shuffled = seqs.clone()
    batch_size, max_len, feat_dim = seqs.shape
    for i in range(batch_size):
        l = lengths[i]
        if l > 1:
            n_to_shuffle = max(1, int(shuffle_fraction * l))
            indices = torch.randperm(l)[:n_to_shuffle]
            shuffled_part = shuffled[i, indices][torch.randperm(n_to_shuffle)]
            shuffled[i, indices] = shuffled_part
    return shuffled

def reverse_sequence(seqs, lengths):
    # [B, T, D] → reversed in T dim
    reversed_seqs = []
    for i, l in enumerate(lengths):
        reversed = torch.flip(seqs[i, :l], dims=[0])
        pad_len = seqs.shape[1] - l
        if pad_len > 0:
            pad = torch.zeros(pad_len, seqs.shape[2], device=seqs.device)
            reversed = torch.cat([reversed, pad], dim=0)
        reversed_seqs.append(reversed)
    return torch.stack(reversed_seqs, dim=0)


def collate_fn(batch):
    sequences, labels, vid_ids = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)  # [B, T_max, D]
    labels = torch.tensor(labels)
    return sequences, lengths, labels, vid_ids

import torch
from torch import nn

class OrthogonalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        seq_len = 33
        # Generate random orthogonal matrix [seq_len, d_model]
        rand = torch.randn(seq_len, d_model)
        q, _ = torch.linalg.qr(rand)
        self.pe = nn.Parameter(q.unsqueeze(0), requires_grad=False)  # Not learnable, fixed

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 33):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class TemporalDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x: [B, T, D]
        B, T, D = x.shape
        # Create mask: [B, T], True means keep, False means drop
        keep_mask = (torch.rand(B, T, device=x.device) > self.p).float().unsqueeze(-1)  # [B, T, 1]
        # Zero-out whole frames (or could rescale for expected sum preservation)
        return x * keep_mask

# class TemporalTransformer(nn.Module):
#     def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, latent_dim, batch_first=True)
#         self.seq_len = 33

#     def forward(self, x, lengths=None):
#         # x: [B, T, input_dim]
#         out, h = self.gru(x)    # h: [1, B, latent_dim]
#         x_out = h[-1]           # [B, latent_dim]
#         x_out = nn.functional.normalize(x_out, p=2, dim=-1)
#         return x_out, None      # No frame embeddings for simplicity

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, d_model),
        )
        self.positional = SinusoidalPositionalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        self.proj = nn.Linear(d_model, latent_dim)

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
        frame_embeddings = x

        # CLS token output at position 0
        cls_emb = x[:, 0, :]                         # [B, D]
        x_out = self.proj(cls_emb)                   # [B, latent_dim]
        x_out = nn.functional.normalize(x_out, p=2, dim=-1)
        return x_out, frame_embeddings


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

def test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE):
    import copy

    results = {}

    model.eval()
    with torch.no_grad():
        # Baseline
        base_embeds, base_labels = [], []
        for seqs, lengths, labels, vid_ids in test_loader:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(seqs, lengths)
            base_embeds.append(emb.cpu())
            base_labels.append(labels)
        base_embeds = torch.cat(base_embeds)
        base_labels = torch.cat(base_labels)
        results["original"] = (base_embeds, base_labels)

        # Shuffled frames
        shuf_embeds, shuf_labels = [], []
        for seqs, lengths, labels, vid_ids in test_loader:
            shuf_seqs = seqs.clone()
            shuf_seqs = partial_shuffle_within_window(shuf_seqs, lengths, vid_ids)
            shuf_seqs, lengths = shuf_seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(shuf_seqs, lengths)
            shuf_embeds.append(emb.cpu())
            shuf_labels.append(labels)
        shuf_embeds = torch.cat(shuf_embeds)
        shuf_labels = torch.cat(shuf_labels)
        results["shuffled"] = (shuf_embeds, shuf_labels)

        # After getting base_embeds and shuf_embeds:
        diff = (base_embeds - shuf_embeds).norm(dim=1)
        print("Mean difference between original and shuffled:", diff.mean().item())
        # exit()

        # Reversed frames
        rev_embeds, rev_labels = [], []
        for seqs, lengths, labels, vid_ids in test_loader:
            rev_seqs = seqs.clone()
            rev_seqs = reverse_sequence(rev_seqs, lengths)
            rev_seqs, lengths = rev_seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(rev_seqs, lengths)
            rev_embeds.append(emb.cpu())
            rev_labels.append(labels)
        rev_embeds = torch.cat(rev_embeds)
        rev_labels = torch.cat(rev_labels)
        results["reversed"] = (rev_embeds, rev_labels)

    # Now, for each version, you can compute consistency or centroid distance
    for key in results:
        emb, labels = results[key]
        # --- Compute distances to centroids as you do for the original test set ---
        per_class_same_dists = {cls: [] for cls in centroids}
        per_class_to_other_dists = {cls: {other: [] for other in centroids if other != cls} for cls in centroids}

        for e, l in zip(emb, labels):
            l = int(l.item())
            dist_same = torch.norm(e - centroids[l]).item()
            per_class_same_dists[l].append(dist_same)
            for c in centroids:
                if c != l:
                    dist_other = torch.norm(e - centroids[c]).item()
                    per_class_to_other_dists[l][c].append(dist_other)
        print(f"\nResults for {key.upper()}:")
        for cls in centroids:
            same_dists = per_class_same_dists[cls]
            inter_dists_all = []
            for other_cls in centroids:
                if other_cls == cls:
                    continue
                inter_dists_all.extend(per_class_to_other_dists[cls][other_cls])
            intra_mean = np.mean(same_dists) if same_dists else np.nan
            inter_mean = np.mean(inter_dists_all) if inter_dists_all else np.nan
            score = inter_mean / (inter_mean + intra_mean) if (not np.isnan(intra_mean) and not np.isnan(inter_mean) and (intra_mean + inter_mean) > 0) else float('nan')
            print(f"  {ALL_CLASSES[cls]}: Consistency Score = {score:.4f}")

    return results

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
    print("✅ Loading datasets...")

    # if os.path.exists(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt"):
    #     train_samples = torch.load(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    #     train_labels = torch.load(f"SAVE_NEW/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    #     test_samples = torch.load(f"SAVE_NEW/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    #     test_labels = torch.load(f"SAVE_NEW/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    #     train_vid_ids = torch.load(f"SAVE_NEW/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    #     test_vid_ids = torch.load(f"SAVE_NEW/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    #     class PoseVideoDatasetFromTensors(Dataset):
    #         def __init__(self, samples, labels, vid_ids):
    #             self.samples = torch.stack(samples)
    #             self.labels = torch.tensor(labels)
    #             self.vid_ids = vid_ids

    #         def __len__(self):
    #             return len(self.samples)

    #         def __getitem__(self, idx):
    #             return self.samples[idx], self.labels[idx], self.vid_ids[idx]
    #     train_dataset = PoseVideoDatasetFromTensors(train_samples, train_labels, train_vid_ids)
    #     test_dataset = PoseVideoDatasetFromTensors(test_samples, test_labels, test_vid_ids)
    #     print(f"✅ Loaded existing datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")
    #     print(f"✅ Train samples: {train_dataset.samples.shape}, Train labels: {train_dataset.labels.shape}, Train vid_ids: {len(train_dataset.vid_ids)}")
    #     print(f"✅ Test samples: {test_dataset.samples.shape}, Test labels: {test_dataset.labels.shape}, Test vid_ids: {len(test_dataset.vid_ids)}")
    #     print("✅ Using existing datasets for training...")
    # else:
    print("❗ No existing datasets found, creating new ones...")
    train_dataset = PoseVideoDataset(
        REAL_ROOT,
        ALL_CLASSES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split="train"
    )
    test_dataset = PoseVideoDataset(
        REAL_ROOT,
        ALL_CLASSES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split="test"
    )

    # save dataset as tensors
    torch.save(train_dataset.samples, f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(train_dataset.labels, f"SAVE_NEW/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.samples, f"SAVE_NEW/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.labels, f"SAVE_NEW/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(train_dataset.vid_ids, f"SAVE_NEW/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.vid_ids, f"SAVE_NEW/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    print(f"✅ Created new datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")

    exit()
        

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    sampler = EnsurePositivesSampler(train_labels, batch_size=BATCH_SIZE, min_pos_per_class=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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

    print("✅ Building model...")
    input_dim = train_dataset[0][0].shape[-1]
    model = TemporalTransformer(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)

    # print number of parameters
    print(f"✅ Number of parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = TCL()
    loss_hard = SupConWithHardNegatives()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_iter = len(train_loader) * EPOCHS
    print(f"✅ Total iterations: {num_iter}")
    cosine_annel = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-6)

    loss_dict = {'loss': [], 'loss_con': [], 'loss_hard': []}

    print("✅ Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_loss_con = 0
        total_loss_hard = 0
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
        #         save_dir="SAVE_NEW/tsne_joint"
        #     )
        #     model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            seqs, lengths, labels, vid_ids = batch
            seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            shuffled_seqs = partial_shuffle_within_window(seqs, lengths, vid_ids)
            optimizer.zero_grad()
            embeddings, frame_embeddings = model(seqs, lengths)
            shuffled_embeddings, _ = model(shuffled_seqs, lengths)

            reverse_seqs = reverse_sequence(seqs, lengths)
            reverse_embeddings, _ = model(reverse_seqs, lengths)

            # get cosine sim 
            cosine_sim = torch.cosine_similarity(embeddings.unsqueeze(1), shuffled_embeddings.unsqueeze(0), dim=-1)  # [B, B]         
            loss_org = loss_fn(embeddings, labels)

            # # ——— temporal smoothness penalty ———
            # # build a map: vid_id → list of batch‐indices
            vid_to_indices = defaultdict(list)
            for i, vid in enumerate(vid_ids):
                vid_to_indices[vid].append(i)


            loss_hard_combined = loss_hard(embeddings, embeddings, shuffled_embeddings)  + loss_hard(embeddings, embeddings, reverse_embeddings)

            print(cosine_sim.mean().item(), loss_org.item(), loss_hard_combined.item())

            loss = loss_org +  10 * loss_hard_combined  # Combine original loss and hard negative loss
            # loss = loss_org
            # loss = loss_org + 10 * loss_hard_combined
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_con += loss_org.item()
            total_loss_hard += loss_hard_combined.item()
            cosine_annel.step()
        print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        loss_dict['loss'].append(total_loss / len(train_loader))
        loss_dict['loss_con'].append(total_loss_con / len(train_loader))
        loss_dict['loss_hard'].append(total_loss_hard / len(train_loader))
        # if (epoch) % 10 == 0 or epoch == 0:
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
        #         save_dir="SAVE_NEW/tsne_joint"
        #     )
        #     model.train()

    print("✅ Saving model...")
    torch.save(model.state_dict(), f"SAVE_NEW/temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    # plot all loss curves, each loss as a subplot
    fig, axs = plt.subplots(3, figsize=(10, 12))
    axs[0].plot(loss_dict['loss'], label='Total Loss')
    axs[0].set_title('Total Loss')
    axs[0].legend()

    axs[1].plot(loss_dict['loss_con'], label='Contrastive Loss')
    axs[1].set_title('Contrastive Loss')
    axs[1].legend()

    axs[2].plot(loss_dict['loss_hard'], label='Hard Loss')
    axs[2].set_title('Hard Loss')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"SAVE_NEW/loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")
    print(f"✅ Saved as loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    print("✅ Computing train embeddings...")
    model.eval()
    all_train_embeds = []
    all_train_labels = []
    with torch.no_grad():
        for seqs, lengths, labels, vid_ids in tqdm(train_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(seqs, lengths)
            all_train_embeds.append(emb.cpu())
            all_train_labels.append(labels)
    all_train_embeds = torch.cat(all_train_embeds)
    all_train_labels = torch.cat(all_train_labels)

    # save all_train_embeds and all_train_labels
    torch.save(all_train_embeds, f"SAVE_NEW/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(all_train_labels, f"SAVE_NEW/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    # Compute class centroids
    centroids = {}
    for cls in torch.unique(all_train_labels):
        mask = all_train_labels == cls
        centroid = all_train_embeds[mask].mean(dim=0)
        centroids[int(cls.item())] = centroid

    print("✅ Evaluating on test set...")
    model.eval()

    # Collect distances per (true_class, other_class)
    per_class_same_dists = {cls: [] for cls in centroids}
    per_class_to_other_dists = {cls: {other: [] for other in centroids if other != cls} for cls in centroids}

    with torch.no_grad():
        for seqs, lengths, labels, vid_ids in tqdm(test_loader):
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
    print("\n✅ Per-Class Distance Statistics:")

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

    print("\n✅ Consistency Scores:")

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

    print("\n✅ Overall Class Consistency Scores:")
    pprint({ALL_CLASSES[k]: v for k, v in consistency_scores.items()})

    with open(f"SAVE_NEW/centroids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pkl", "wb") as f:
        pickle.dump(centroids, f)

    print("\n✅ Done!")

    print("\n✅ Visualizing window embeddings and centroids...")

    # Compute test embeddings
    print("✅ Computing test embeddings...")
    all_test_embeds = []
    all_test_labels = []
    with torch.no_grad():
        for seqs, lengths, labels, vid_ids in tqdm(test_loader):
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
    plt.savefig(f"SAVE_NEW/embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png", dpi=200)
    print(f"✅ Saved as embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    results = test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE)

# ——— MAIN ———
if __name__ == "__main__":
    train()