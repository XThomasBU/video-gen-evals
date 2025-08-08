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

def second_order_steady_loss(frame_embs):
    """
    frame_embs: [B, T, D] (no CLS token)
    Returns scalar steady loss for the batch.
    """
    # Compute pairwise differences
    diff1 = frame_embs[:, :-2, :] - frame_embs[:, 1:-1, :]   # [B, T-2, D]
    diff2 = frame_embs[:, 1:-1, :] - frame_embs[:, 2:, :]    # [B, T-2, D]
    # Difference of differences (second order derivative)
    steady = diff1 - diff2                                   # [B, T-2, D]
    loss = (steady ** 2).mean()                              # MSE
    return loss

from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

def plot_embeddings_epoch(epoch, all_train_embeds, all_train_labels, all_test_embeds, all_test_labels, ALL_CLASSES, save_dir="SAVE_NEW2/tsne_joint"):
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)
    train_embeds = all_train_embeds.cpu().numpy()
    train_labels = all_train_labels.cpu().numpy()
    test_embeds = all_test_embeds.cpu().numpy()
    test_labels = all_test_labels.cpu().numpy()

    # Compute centroids for each class (train set only)
    centroids = []
    for cls in range(len(ALL_CLASSES)):
        mask = train_labels == cls
        centroids.append(train_embeds[mask].mean(axis=0))
    centroids = np.stack(centroids)

    # Stack everything for joint t-SNE
    combined_embeds = np.concatenate([train_embeds, test_embeds, centroids], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=3000, init='pca')
    projected_all = tsne.fit_transform(combined_embeds)

    n_train = len(train_embeds)
    n_test = len(test_embeds)
    n_classes = len(ALL_CLASSES)
    projected_train = projected_all[:n_train]
    projected_test = projected_all[n_train:n_train+n_test]
    projected_centroids = projected_all[-n_classes:]

    colors = plt.cm.get_cmap("tab10", n_classes)
    plt.figure(figsize=(10, 8))

    # Plot train embeddings
    for cls in range(n_classes):
        mask = train_labels == cls
        plt.scatter(
            projected_train[mask, 0],
            projected_train[mask, 1],
            s=10,
            color=colors(cls),
            label=ALL_CLASSES[cls],
            alpha=0.6
        )

    # Plot centroids
    for i, (x, y) in enumerate(projected_centroids):
        plt.scatter(x, y, color=colors(i), edgecolors='k', s=200, marker='X', linewidths=2)
        plt.text(x, y + 0.05, ALL_CLASSES[i], fontsize=9, ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.8))

    # Plot test embeddings as stars
    for cls in range(n_classes):
        mask = test_labels == cls
        plt.scatter(
            projected_test[mask, 0],
            projected_test[mask, 1],
            s=100,
            facecolors=colors(cls),
            edgecolors='black',
            linewidths=0.5,
            marker='*',
            label=f"{ALL_CLASSES[cls]} (test)" if epoch == 0 else None,  # Avoid double legend
            alpha=0.9
        )

    plt.title(f"t-SNE Train+Test+Centroids, Epoch {epoch+1}")
    # Only show unique legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8, markerscale=1.2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/tsne_train_test_centroids_epoch_{epoch+1:03d}.png", dpi=200)
    plt.close()

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

def get_static_window(seqs):
    # static window -- replace window with the first frame of the sequence
    static_seqs = []
    for seq in seqs:
        first_frame = seq[0].unsqueeze(0)  # [1, D]
        static_seq = first_frame.repeat(seq.shape[0], 1)  # [T, D]
        static_seqs.append(static_seq)
    return torch.stack(static_seqs, dim=0)  # [B, T, D]


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
        frame_embeddings = x

        # CLS token output at position 0
        cls_emb = x[:, 0, :]                         # [B, D]
        x_out = self.proj(cls_emb)                   # [B, latent_dim]
        x_out = nn.functional.normalize(x_out, p=2, dim=-1)

        frame_embeddings = self.frame_proj(frame_embeddings)  # [B, T+1, latent_dim]
        frame_embeddings = nn.functional.normalize(frame_embeddings, p=2, dim=-1)  # Normalize frame embeddings
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


def frame_dropout(seqs, lengths, drop_fraction=0.2):
    # Randomly drop (zero out) a fraction of frames for each sequence
    batch = seqs.clone()
    B, T, D = batch.shape
    for i in range(B):
        l = lengths[i].item()
        n_drop = int(l * drop_fraction)
        if n_drop > 0:
            drop_indices = torch.randperm(l)[:n_drop]
            batch[i, drop_indices] = 0  # Zero out selected frames
    return batch

def frame_repetition(seqs, lengths, repeat_times=2):
    # Repeat each frame repeat_times, then truncate or pad to original length
    B, T, D = seqs.shape
    new_T = T * repeat_times
    repeated = []
    for i in range(B):
        l = lengths[i].item()
        seq = seqs[i, :l]  # [l, D]
        seq_rep = seq.repeat_interleave(repeat_times, dim=0)  # [l*repeat_times, D]
        # Pad or truncate to original T
        if seq_rep.shape[0] < T:
            pad = torch.zeros(T - seq_rep.shape[0], D, device=seqs.device)
            seq_rep = torch.cat([seq_rep, pad], dim=0)
        else:
            seq_rep = seq_rep[:T]
        repeated.append(seq_rep)
    return torch.stack(repeated, dim=0)

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

    # Frame Dropout
    drop_embeds, drop_labels = [], []
    for seqs, lengths, labels, vid_ids in test_loader:
        drop_seqs = frame_dropout(seqs, lengths, drop_fraction=0.3)
        drop_seqs, lengths = drop_seqs.to(DEVICE), lengths.to(DEVICE)
        emb, _ = model(drop_seqs, lengths)
        drop_embeds.append(emb.cpu())
        drop_labels.append(labels)
    drop_embeds = torch.cat(drop_embeds)
    drop_labels = torch.cat(drop_labels)
    results["dropout"] = (drop_embeds, drop_labels)
    diff_drop = (base_embeds - drop_embeds).norm(dim=1)
    print("Mean difference between original and dropout:", diff_drop.mean().item())

    # Frame Repetition
    rep_embeds, rep_labels = [], []
    for seqs, lengths, labels, vid_ids in test_loader:
        rep_seqs = frame_repetition(seqs, lengths, repeat_times=2)
        rep_seqs, lengths = rep_seqs.to(DEVICE), lengths.to(DEVICE)
        emb, _ = model(rep_seqs, lengths)
        rep_embeds.append(emb.cpu())
        rep_labels.append(labels)
    rep_embeds = torch.cat(rep_embeds)
    rep_labels = torch.cat(rep_labels)
    results["repetition"] = (rep_embeds, rep_labels)
    diff_rep = (base_embeds - rep_embeds).norm(dim=1)
    print("Mean difference between original and repetition:", diff_rep.mean().item())


    # Static Window
    static_emb, static_labels = [], []
    for seqs, lengths, labels, vid_ids in test_loader:
        static_seqs = get_static_window(seqs)
        static_seqs, lengths = static_seqs.to(DEVICE), lengths.to(DEVICE)
        emb, _ = model(static_seqs, lengths)
        static_emb.append(emb.cpu())
        static_labels.append(labels)
    static_emb = torch.cat(static_emb)
    static_labels = torch.cat(static_labels)
    results["static"] = (static_emb, static_labels)
    diff_static = (base_embeds - static_emb).norm(dim=1)
    print("Mean difference between original and static window:", diff_static.mean().item())

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

    if os.path.exists(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt"):
        # train_samples = torch.load(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        # train_labels = torch.load(f"SAVE_NEW/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        # test_samples = torch.load(f"SAVE_NEW/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        # test_labels = torch.load(f"SAVE_NEW/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        # train_vid_ids = torch.load(f"SAVE_NEW/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        # test_vid_ids = torch.load(f"SAVE_NEW/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_samples = torch.load(f"SAVE_NEW/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_labels = torch.load(f"SAVE_NEW/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_samples = torch.load(f"SAVE_NEW/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_labels = torch.load(f"SAVE_NEW/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_vid_ids = torch.load(f"SAVE_NEW/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_vid_ids = torch.load(f"SAVE_NEW/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

        class PoseVideoDatasetFromTensors(Dataset):
            def __init__(self, samples, labels, vid_ids):
                self.samples = torch.stack(samples)
                self.labels = torch.tensor(labels)
                self.vid_ids = vid_ids

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx], self.labels[idx], self.vid_ids[idx]
        train_dataset = PoseVideoDatasetFromTensors(train_samples, train_labels, train_vid_ids)
        test_dataset = PoseVideoDatasetFromTensors(test_samples, test_labels, test_vid_ids)
        print(f"✅ Loaded existing datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")
        print(f"✅ Train samples: {train_dataset.samples.shape}, Train labels: {train_dataset.labels.shape}, Train vid_ids: {len(train_dataset.vid_ids)}")
        print(f"✅ Test samples: {test_dataset.samples.shape}, Test labels: {test_dataset.labels.shape}, Test vid_ids: {len(test_dataset.vid_ids)}")
        print("✅ Using existing datasets for training...")
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_iter = len(train_loader) * EPOCHS
    print(f"✅ Total iterations: {num_iter}")
    cosine_annel = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-6)

    loss_dict = {'loss': [], 'loss_con': [], 'loss_hard': [], 'smoothness': [], 'hard_shuffle_big': [], 'hard_shuffle_small': [], 'reverse': [], 'static': []}

    print("✅ Starting training...")
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

            shuffled_seqs_small = partial_shuffle_within_window(seqs, lengths, vid_ids, shuffle_fraction=0.3)
            shuffled_embeddings_small, _ = model(shuffled_seqs_small, lengths)

            # # ——— temporal smoothness penalty ———
            # # build a map: vid_id → list of batch‐indices
            vid_to_indices = defaultdict(list)
            for i, vid in enumerate(vid_ids):
                vid_to_indices[vid].append(i)

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
            frame_smoothness_loss = second_order_steady_loss(frame_embeddings[:, 1:])  # exclude CLS toke

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
        print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
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

    print("✅ Saving model...")
    torch.save(model.state_dict(), f"SAVE_NEW2/temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

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
    print(f"✅ Saved as loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    print("✅ Computing train embeddings...")
    model.eval()
    all_train_embeds = []
    all_train_labels = []
    all_train_vid_ids = []   # <-- NEW
    with torch.no_grad():
        for seqs, lengths, labels, vid_ids in tqdm(train_loader):
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

    with open(f"SAVE_NEW2/centroids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pkl", "wb") as f:
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
    plt.savefig(f"SAVE_NEW2/embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png", dpi=200)
    print(f"✅ Saved as embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    results = test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE)

# ——— MAIN ———
if __name__ == "__main__":
    train()