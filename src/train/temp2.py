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
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 100
WINDOW_SIZE = 64 # 64
STRIDE = 8 # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    for p in frames:
        try:
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

            # # Normalize each part
            # vit_feature   /= np.linalg.norm(vit_feature) + 1e-8
            # global_orient /= np.linalg.norm(global_orient) + 1e-8
            # body_pose     /= np.linalg.norm(body_pose) + 1e-8
            # betas         /= np.linalg.norm(betas) + 1e-8

            vec = np.concatenate([vit_feature, global_orient, body_pose, betas], axis=0)
            vec = vec / np.linalg.norm(vec) + 1e-8
            if vec.shape[0] != 1250:
                continue

            frame_vecs.append(torch.tensor(vec, dtype=torch.float32))
        except:
            continue

    if len(frame_vecs) < 2:
        return None

    # [T, 1250]
    frame_tensor = torch.stack(frame_vecs, dim=0)

    # Compute motion vectors (frame-to-frame deltas)
    motion_vecs = frame_tensor[1:] - frame_tensor[:-1]  # [T-1, 1250]
    motion_vecs = torch.cat([torch.zeros(1, 1250), motion_vecs], dim=0)  # [T, 1250]

    # Concatenate original + motion
    enriched_tensor = torch.cat([frame_tensor, motion_vecs], dim=1)  # [T, 2500]

    # return enriched_tensor
    return frame_tensor

# ——— SLIDING WINDOW ———
def extract_windows(seq, window_size, stride):
    windows = []
    num_frames = seq.shape[0]
    for start in range(0, num_frames, stride):
        end = start + window_size
        if end > num_frames:
            pad_len = end - num_frames
            pad = seq[-1:].repeat(pad_len, 1)
            window = torch.cat([seq[start:], pad], dim=0)
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
        for cls in classes:
            videos = self.video_split[cls]
            for vid_path in videos:
                seq = load_video_sequence(vid_path)
                if seq is None:
                    continue
                windows = extract_windows(seq, WINDOW_SIZE, STRIDE)
                self.samples.extend(windows)
                self.labels.extend([self.class_to_idx[cls]] * len(windows))
                self.vid_ids.extend([vid_path.name] * len(windows))

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

        self.attn_query = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, d_model]
        self.proj = nn.Linear(d_model, latent_dim)

    def forward(self, x, lengths):
        x = self.input_proj(x)  # [B, T, d_model]
        B, T, D = x.shape

        # pos_emb = self.positional[:T, :].unsqueeze(0)
        # x = x + pos_emb
        x = self.positional(x)

        # Mask: True where padding is applied
        mask = torch.arange(T, device=lengths.device)[None, :] >= lengths[:, None]
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, T, D]
        frame_embeddings = x

        # Attention pooling: Q = [1, 1, D], K,V = [B, T, D] → out = [B, 1, D]
        q = self.attn_query.expand(B, -1, -1)  # [B, 1, D]
        attn_weights = torch.softmax((q @ x.transpose(1, 2)) / (D ** 0.5), dim=-1)  # [B, 1, T]
        pooled = attn_weights @ x  # [B, 1, D]
        pooled = pooled.squeeze(1)  # [B, D]

        x = self.proj(pooled)
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x, frame_embeddings

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

    print("✅ Building model...")
    input_dim = train_dataset[0][0].shape[-1]
    model = TemporalTransformer(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)

    # print number of parameters
    print(f"✅ Number of parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = SupConLoss()
    loss_fn_hard = SupConWithHardNegatives()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("✅ Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            seqs, lengths, labels, vid_ids = batch
            seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            shuffled_seqs = partial_shuffle_within_window(seqs, lengths, vid_ids)
            optimizer.zero_grad()
            embeddings, frame_embeddings = model(seqs, lengths)
            shuffled_embeddings, _ = model(shuffled_seqs, lengths)
            positive = embeddings.detach()
            # hard_negative = shuffled_embeddings.detach()
            # loss_hard = loss_fn_hard(embeddings, positive, hard_negative)
            # loss = loss_fn(embeddings, labels)
            # print(loss, loss_hard)

            # # ——— temporal smoothness penalty ———
            # # build a map: vid_id → list of batch‐indices
            vid_to_indices = defaultdict(list)
            for i, vid in enumerate(vid_ids):
                vid_to_indices[vid].append(i)

            curvature_loss = 0.0
            for vid, idxs in vid_to_indices.items():
                if len(idxs) < 3:
                    continue
                idxs = sorted(idxs)
                z_seq = frame_embeddings[idxs]           # [T, D]
                vel = z_seq[1:] - z_seq[:-1]            # [T-1, D]
                acc = vel[1:] - vel[:-1]                # [T-2, D]
                acc_loss = acc.pow(2).sum(dim=-1).mean()  # Mean over all frames in this video
                curvature_loss += acc_loss
            curvature_loss = curvature_loss / max(1, len(vid_to_indices))

            rev_seqs = reverse_sequence(seqs, lengths)
            reversed_embeds, _ = model(rev_seqs, lengths)

            # # Encourage dissimilarity between embeddings and their reversed counterparts
            # rev_cos_sim = torch.sum(embeddings * reversed_embeds, dim=-1)  # [B]
            # rev_loss = rev_cos_sim.mean()  # higher = too similar → penalize
            # print(rev_loss)\
            loss_org = loss_fn(embeddings, labels)
            loss_hard_combined = loss_fn_hard(embeddings, positive, shuffled_embeddings.detach()) + \
                     loss_fn_hard(embeddings, positive, reversed_embeds.detach())


            loss = loss_org + 10 * loss_hard_combined 
            print(loss_org, loss_hard_combined, curvature_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    print("✅ Saving model...")
    torch.save(model.state_dict(), f"temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}.pt")

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
    torch.save(all_train_embeds, f"all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}.pt")
    torch.save(all_train_labels, f"all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}.pt")

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

    with open(f"centroids_window_{WINDOW_SIZE}_stride_{STRIDE}.pkl", "wb") as f:
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
    plt.savefig(f"SAVE/embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}.png", dpi=200)
    print(f"✅ Saved as embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}.png")

# ——— MAIN ———
if __name__ == "__main__":
    train()