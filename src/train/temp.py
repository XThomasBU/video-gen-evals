# import numpy as np
# import pickle
# import os
# from pathlib import Path
# from tqdm import tqdm
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist, jensenshannon

# REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"

# # ——— HELPERS ———
# def load_data(video_paths):
#     all_videos = {}
#     for video_path in video_paths:
#         frames = sorted(Path(video_path).glob("tokenhmr_mesh/*.pkl"))
#         all_out = {}
#         for p in frames:
#             with open(p, "rb") as f:
#                 data = pickle.load(f)
#             all_out[p.stem] = data
#         all_videos[Path(video_path).stem] = all_out
#     return all_videos


# def filter_data(data):
#     filtered_data = {}
#     for video_path, video_data in data.items():
#         filtered_data[video_path] = {}
#         for frame, info in video_data.items():
#             global_orient = info['pred_smpl_params']['global_orient']
#             body_pose = info['pred_smpl_params']['body_pose']
#             betas = info['pred_smpl_params']['betas']
#             global_orient = np.array(global_orient).flatten()
#             body_pose = np.array(body_pose).flatten()
#             betas = np.array(betas).flatten()
#             filtered_data[video_path][frame] = np.concatenate([global_orient, body_pose, betas], axis=0).flatten()
#     return filtered_data
       
# def compute_bin_sequence(token_seq, token_to_bin):
#     """Convert frame-by-frame token IDs to bin IDs"""
#     return [[token_to_bin[t] for t in frame] for frame in token_seq]

# def extract_ngrams_from_sequence(bin_seq, ngram_size, n_bins):
#     """
#     bin_seq: list of frames, each frame is list of bin ids
#     Returns: flattened histogram over n-grams
#     """
#     ngram_counts = {}
#     for f in range(len(bin_seq) - ngram_size + 1):
#         for pos in zip(*[bin_seq[f+i] for i in range(ngram_size)]):
#             key = tuple(pos)
#             ngram_counts[key] = ngram_counts.get(key, 0) + 1

#     total_count = sum(ngram_counts.values())
#     hist_size = n_bins ** ngram_size
#     hist = np.zeros(hist_size, dtype=np.float32)
#     for key, count in ngram_counts.items():
#         idx = 0
#         for i, b in enumerate(key):
#             idx += b * (n_bins ** (ngram_size - i - 1))
#         hist[idx] = count / total_count
#     return hist

# # ——— MAIN FLOW ———
# def main():
#     class_name = "JumpingJack"
#     print(f"\n✅ Target class: {class_name}")

#     # ——— Load REAL videos ———
#     print("\n✅ Loading REAL videos...")
#     real_class_dir = Path(REAL_ROOT) / class_name

#     real_videos = os.listdir(real_class_dir)
#     real_video_paths = [f"{real_class_dir}/{v}" for v in real_videos]

#     real_data = load_data(real_video_paths)
#     filtered_data = filter_data(real_data)


# if __name__ == "__main__":
#     main()


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

# ——— CONFIG ———
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
BATCH_SIZE = 256
LATENT_DIM = 256
EPOCHS = 100
WINDOW_SIZE = 64 # 64
STRIDE = 16 # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ——— LOAD VIDEO FRAMES ———
def load_video_sequence(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []
    for p in frames:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            params = data["pred_smpl_params"]

            # ✅ Multi-person case: take first person
            if isinstance(params, list):
                if len(params) < 1:
                    continue
                params = params[0]

            if not isinstance(params, dict):
                continue

            vit_feature = np.array(params["token_out"]).flatten()

            global_orient = np.array(params["global_orient"]).flatten()
            body_pose = np.array(params["body_pose"]).flatten()
            betas = np.array(params["betas"]).flatten()

            vec = np.concatenate([vit_feature, global_orient, body_pose, betas], axis=0)
            if vec.shape[0] != 1250:
                continue
            # vec = np.concatenate([global_orient, body_pose, betas], axis=0)
            # if vec.shape[0] != 226:
            #     continue

            frame_vecs.append(torch.tensor(vec, dtype=torch.float32))
        except:
            continue

    if len(frame_vecs) == 0:
        return None

    return torch.stack(frame_vecs)

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

# ——— DATASET ———
class PoseVideoDataset(Dataset):
    def __init__(self, root, classes, window_size=64, stride=32, split="train"):
        self.samples = []
        self.labels = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.per_class_data = {c: [] for c in classes}
        for cls in classes:
            class_dir = Path(root) / cls
            for vid in os.listdir(class_dir):
                vid_path = class_dir / vid
                seq = load_video_sequence(vid_path)
                if seq is None:
                    continue
                windows = extract_windows(seq, window_size, stride)
                self.per_class_data[cls].extend(windows)

        # Train/test split
        self.samples = []
        self.labels = []
        for cls in classes:
            all_windows = self.per_class_data[cls]
            np.random.shuffle(all_windows)
            n_train = int(0.8 * len(all_windows))
            if split == "train":
                selected = all_windows[:n_train]
            else:
                selected = all_windows[n_train:]
            self.samples.extend(selected)
            self.labels.extend([self.class_to_idx[cls]] * len(selected))

        print(f"✅ Loaded {len(self.samples)} {split} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# ——— COLLATE FN ———
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences)
    labels = torch.tensor(labels)
    lengths = torch.full((len(labels),), sequences.shape[1], dtype=torch.long)
    return sequences, lengths, labels

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
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, d_model),
        )

        self.positional = nn.Parameter(torch.randn(512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, latent_dim)

    def forward(self, x, lengths):
        # Project to d_model
        x = self.input_proj(x)  # Shape: [batch, seq, d_model]
        batch_size, max_len, _ = x.shape

        # Add positional encoding
        pos_emb = self.positional[:max_len, :].unsqueeze(0)
        x = x + pos_emb

        # Mask
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
        x = self.transformer(x, src_key_padding_mask=mask)

        # Masked mean-pooling
        mask_float = (~mask).unsqueeze(-1).float()
        summed = (x * mask_float).sum(dim=1)
        counts = mask_float.sum(dim=1).clamp(min=1e-6)
        x = summed / counts

        x = self.proj(x)
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

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
    loss_fn = SupConLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("✅ Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            seqs, lengths, labels = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            embeddings = model(seqs, lengths)
            loss = loss_fn(embeddings, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    print("✅ Saving model...")
    torch.save(model.state_dict(), "temporal_transformer_model.pt")

    print("✅ Computing train embeddings...")
    model.eval()
    all_train_embeds = []
    all_train_labels = []
    with torch.no_grad():
        for seqs, lengths, labels in tqdm(train_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb = model(seqs, lengths)
            all_train_embeds.append(emb.cpu())
            all_train_labels.append(labels)
    all_train_embeds = torch.cat(all_train_embeds)
    all_train_labels = torch.cat(all_train_labels)

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
        for seqs, lengths, labels in tqdm(test_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb = model(seqs, lengths).cpu()
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

    print("\n✅ Done!")

# ——— MAIN ———
if __name__ == "__main__":
    train()