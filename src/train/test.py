import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

from temp2 import (
    TemporalTransformer,
    load_video_sequence,
    extract_windows,
    collate_fn,
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)

GEN_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos"
BATCH_SIZE = 64

# ————— load train embeddings & centroids —————
all_train_embeds = torch.load("all_train_embeds.pt").numpy()
all_train_labels = torch.load("all_train_labels.pt").numpy()

with open("centroids.pkl", "rb") as f:
    centroids = pickle.load(f)
for k in centroids:
    centroids[k] = centroids[k].to(DEVICE)

# ————— load model —————
print("✅ Loading model...")
model = TemporalTransformer(input_dim=1250, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load("temporal_transformer_model.pt"))
model.eval()

# ————— generated dataset & loader —————
class GeneratedVideoDataset(Dataset):
    def __init__(self, root, classes, window_size=WINDOW_SIZE, stride=STRIDE):
        self.samples, self.labels, self.video_names = [], [], []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            for vid in os.listdir(Path(root) / cls):
                seq = load_video_sequence(Path(root) / cls / vid)
                if seq is None:
                    continue
                wins = extract_windows(seq, window_size, stride)
                self.samples.extend(wins)
                self.labels.extend([self.class_to_idx[cls]] * len(wins))
                self.video_names.extend([vid] * len(wins))
        print(f"✅ Loaded {len(self.samples)} windows from generated videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.video_names[idx]

gen_dataset = GeneratedVideoDataset(GEN_ROOT, ALL_CLASSES)
gen_loader = DataLoader(
    gen_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=lambda x: (
        torch.stack([i[0] for i in x]),
        torch.tensor([i[1] for i in x]),
        [i[2] for i in x],
    ),
)

# ————— evaluate & collect per-window scores & embeddings —————
print("\n✅ Evaluating generated videos...")
video_results = {}
video_embeds = defaultdict(list)
video_labels = {}

with torch.no_grad():
    for seqs, labels, vid_names in tqdm(gen_loader):
        seqs = seqs.to(DEVICE)
        lengths = torch.full((seqs.shape[0],), seqs.shape[1], dtype=torch.long, device=DEVICE)
        embs = model(seqs, lengths).cpu()

        for emb, lbl, vid in zip(embs, labels, vid_names):
            intra = torch.norm(emb - centroids[int(lbl)].cpu()).item()
            inter = [
                torch.norm(emb - centroids[c].cpu()).item()
                for c in centroids
                if c != int(lbl)
            ]
            inter_mean = np.mean(inter)
            score = float(inter_mean / (inter_mean + intra)) if (intra + inter_mean) != 0 else np.nan

            # store per-window metrics
            entry = video_results.setdefault(vid, {"cls": int(lbl), "intra": [], "consistency": []})
            entry["intra"].append(intra)
            entry["consistency"].append(score)

            # also collect for embedding visualization
            video_embeds[vid].append(emb)
            video_labels[vid] = int(lbl)

# ————— rank by outlier-aware (weighted) averaging —————
video_scores = []
for vid, vals in video_results.items():
    # mean intra-distance
    mean_intra = np.mean(vals["intra"])

    # outlier-aware weighted average of consistency scores
    scores = np.array(vals["consistency"])
    mu, sigma = scores.mean(), scores.std() if scores.std() > 0 else 1e-6
    weights = np.exp(-((scores - mu) ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()
    weighted_consistency = float((weights * scores).sum())

    video_scores.append((vid, ALL_CLASSES[vals["cls"]], mean_intra, weighted_consistency))

# sort descending by consistency
video_scores.sort(key=lambda x: -x[3])

print("\n✅ Generated Video Ranking (Best to Worst by Weighted Consistency):")
for rank, (vid, cls_name, intra, cons) in enumerate(video_scores, 1):
    print(f"{rank}. {vid} | Class: {cls_name} | Intra: {intra:.4f} | Weighted Consistency: {cons:.4f}")

# ————— prepare per-video averaged embeddings for t-SNE —————
vid_names = list(video_embeds.keys())
vid_embeds = np.stack([torch.stack(video_embeds[v]).mean(dim=0).numpy() for v in vid_names])
vid_classes = np.array([video_labels[v] for v in vid_names])
centroid_mat = torch.stack([centroids[c].cpu() for c in sorted(centroids)]).numpy()

# ————— t-SNE projection —————
tsne = TSNE(n_components=2, random_state=42)
combined = np.concatenate([vid_embeds, centroid_mat], axis=0)
proj = tsne.fit_transform(combined)
projected_vid_embeds = proj[: len(vid_embeds)]
projected_centroids = proj[len(vid_embeds) :]

# ————— plot —————
colors = plt.cm.get_cmap("tab10", len(ALL_CLASSES))
plt.figure(figsize=(10, 8))

for cls in range(len(ALL_CLASSES)):
    mask = vid_classes == cls
    plt.scatter(
        projected_vid_embeds[mask, 0],
        projected_vid_embeds[mask, 1],
        s=40,
        color=colors(cls),
        label=f"{ALL_CLASSES[cls]} (gen)",
        alpha=0.9,
    )
for i, (x, y) in enumerate(projected_centroids):
    plt.scatter(x, y, edgecolors="k", s=200, marker="X", linewidths=2)
    plt.text(x, y + 0.2, ALL_CLASSES[i], ha="center", va="bottom", fontsize=9, weight="bold")

plt.title("Per-Video Averaged Embeddings + Centroids (t-SNE Projection)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("gen_embeddings_centroids_per_video.png", dpi=200)
print("✅ Saved as gen_embeddings_centroids_per_video.png")