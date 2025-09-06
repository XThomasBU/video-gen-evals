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
from train import (
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)
from models import TemporalTransformerV2Plus
from utils import *

# set seed
torch.manual_seed(1)
np.random.seed(1)

import types

def patch_transformer_encoder_layers(model):
    for i, layer in enumerate(model.transformer.layers):
        def new_forward(self, src, src_mask=None, src_key_padding_mask=None, *args, **kwargs):
            # Accept any keyword args for forward compatibility
            src2, attn_weights = self.self_attn(
                src, src, src, attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False
            )
            self._saved_attn_weights = attn_weights.detach().cpu()
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src
        layer.forward = types.MethodType(new_forward, layer)

# ——— CONFIG ———
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
INPUT_DIM = 2740
WINDOW_SIZE = 32  # 64
STRIDE = 8  # 32

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

def frame_downsample(seqs, lengths, interval=2):
    """
    Downsample sequences by keeping every `interval`-th frame up to sequence length.
    Args:
        seqs: Tensor of shape (B, T, D)
        lengths: Tensor of shape (B,) with valid sequence lengths
        interval: int, keep every `interval`-th frame
    Returns:
        downsampled_seqs: list of length B, each item is [n_kept_frames, D]
        new_lengths: tensor of kept lengths for each sequence
    """
    B, T, D = seqs.shape
    output = []
    new_lengths = []
    for i in range(B):
        l = lengths[i].item()
        # Indices to keep: 0, interval, 2*interval, ...
        keep_idx = torch.arange(0, l, interval)
        output.append(seqs[i, keep_idx])
        new_lengths.append(len(keep_idx))
    return output, torch.tensor(new_lengths, device=seqs.device)

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

# def test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE):
#     import copy

#     results = {}

#     model.eval()
#     with torch.no_grad():
#         # Baseline
#         base_embeds, base_labels = [], []
#         for seqs, lengths, labels, vid_ids in test_loader:
#             seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
#             emb, _ = model(seqs, lengths)
#             base_embeds.append(emb.cpu())
#             base_labels.append(labels)
#         base_embeds = torch.cat(base_embeds)
#         base_labels = torch.cat(base_labels)
#         results["original"] = (base_embeds, base_labels)

#         # Shuffled frames
#         shuf_embeds, shuf_labels = [], []
#         for seqs, lengths, labels, vid_ids in test_loader:
#             shuf_seqs = seqs.clone()
#             shuf_seqs = partial_shuffle_within_window(shuf_seqs, lengths, vid_ids, shuffle_fraction=0.7)
#             shuf_seqs, lengths = shuf_seqs.to(DEVICE), lengths.to(DEVICE)
#             emb, _ = model(shuf_seqs, lengths)
#             shuf_embeds.append(emb.cpu())
#             shuf_labels.append(labels)
#         shuf_embeds = torch.cat(shuf_embeds)
#         shuf_labels = torch.cat(shuf_labels)
#         results["shuffled"] = (shuf_embeds, shuf_labels)

#         # After getting base_embeds and shuf_embeds:
#         diff = (base_embeds - shuf_embeds).norm(dim=1)
#         print("Mean difference between original and shuffled:", diff.mean().item())
#         # exit()

#         # Reversed frames
#         rev_embeds, rev_labels = [], []
#         for seqs, lengths, labels, vid_ids in test_loader:
#             rev_seqs = seqs.clone()
#             rev_seqs = reverse_sequence(rev_seqs, lengths)
#             rev_seqs, lengths = rev_seqs.to(DEVICE), lengths.to(DEVICE)
#             emb, _ = model(rev_seqs, lengths)
#             rev_embeds.append(emb.cpu())
#             rev_labels.append(labels)
#         rev_embeds = torch.cat(rev_embeds)
#         rev_labels = torch.cat(rev_labels)
#         results["reversed"] = (rev_embeds, rev_labels)

#     # Frame Dropout
#     drop_embeds, drop_labels = [], []
#     for seqs, lengths, labels, vid_ids in test_loader:
#         drop_seqs, lengths = frame_downsample(seqs, lengths,interval=2)
#         drop_seqs, lengths = drop_seqs.to(DEVICE), lengths.to(DEVICE)
#         emb, _ = model(drop_seqs, lengths)
#         drop_embeds.append(emb.cpu())
#         drop_labels.append(labels)
#     drop_embeds = torch.cat(drop_embeds)
#     drop_labels = torch.cat(drop_labels)
#     results["dropout"] = (drop_embeds, drop_labels)
#     diff_drop = (base_embeds - drop_embeds).norm(dim=1)
#     print("Mean difference between original and dropout:", diff_drop.mean().item())

#     # Frame Repetition
#     rep_embeds, rep_labels = [], []
#     for seqs, lengths, labels, vid_ids in test_loader:
#         rep_seqs = frame_repetition(seqs, lengths, repeat_times=5)
#         rep_seqs, lengths = rep_seqs.to(DEVICE), lengths.to(DEVICE)
#         emb, _ = model(rep_seqs, lengths)
#         rep_embeds.append(emb.cpu())
#         rep_labels.append(labels)
#     rep_embeds = torch.cat(rep_embeds)
#     rep_labels = torch.cat(rep_labels)
#     results["repetition"] = (rep_embeds, rep_labels)
#     diff_rep = (base_embeds - rep_embeds).norm(dim=1)
#     print("Mean difference between original and repetition:", diff_rep.mean().item())

#     # Now, for each version, you can compute consistency or centroid distance
#     for key in results:
#         emb, labels = results[key]
#         # --- Compute distances to centroids as you do for the original test set ---
#         per_class_same_dists = {cls: [] for cls in centroids}
#         per_class_to_other_dists = {cls: {other: [] for other in centroids if other != cls} for cls in centroids}

#         for e, l in zip(emb, labels):
#             l = int(l.item())
#             dist_same = torch.norm(e - centroids[l]).item()
#             per_class_same_dists[l].append(dist_same)
#             for c in centroids:
#                 if c != l:
#                     dist_other = torch.norm(e - centroids[c]).item()
#                     per_class_to_other_dists[l][c].append(dist_other)
#         print(f"\nResults for {key.upper()}:")
#         for cls in centroids:
#             same_dists = per_class_same_dists[cls]
#             inter_dists_all = []
#             for other_cls in centroids:
#                 if other_cls == cls:
#                     continue
#                 inter_dists_all.extend(per_class_to_other_dists[cls][other_cls])
#             intra_mean = np.mean(same_dists) if same_dists else np.nan
#             inter_mean = np.mean(inter_dists_all) if inter_dists_all else np.nan
#             score = inter_mean / (inter_mean + intra_mean) if (not np.isnan(intra_mean) and not np.isnan(inter_mean) and (intra_mean + inter_mean) > 0) else float('nan')
#             print(f"  {ALL_CLASSES[cls]}: Consistency Score = {score:.4f}")

#     return results

def collect_and_plot_consistency_scores(results, centroids, ALL_CLASSES, plot_dir="SAVE_NEW2/consistency_plots"):
    os.makedirs(plot_dir, exist_ok=True)
    all_scores = {}

    # 1. Collect scores
    for key in results:
        emb, labels = results[key]
        # Parse distortion and scale from key
        if "_" in key:
            distortion, scale = key.split("_")
            try:
                scale = float(scale)
            except ValueError:
                scale = int(scale)
        else:
            distortion = key
            scale = None

        per_class_same_dists = {cls: [] for cls in centroids}
        per_class_to_other_dists = {cls: {other: [] for other in centroids if other != cls} for cls in centroids}
        for e, l in zip(emb, labels):
            l = int(l.item())
            dist_same = np.linalg.norm(e.detach().cpu().numpy() - centroids[l].detach().cpu().numpy())
            per_class_same_dists[l].append(dist_same)
            for c in centroids:
                if c != l:
                    dist_other = np.linalg.norm(e.detach().cpu().numpy() - centroids[c].detach().cpu().numpy())
                    per_class_to_other_dists[l][c].append(dist_other)

        # Calculate per-class scores
        scores = {}
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
            scores[cls] = score

        if distortion not in all_scores:
            all_scores[distortion] = {}
        all_scores[distortion][scale] = scores

    # 2. Get original and reversed for reference if exists
    orig_scores = all_scores.get("original", {}).get(None, None)
    # reversed_scores = all_scores.get("reversed", {}).get(None, None)

    # 3. Plotting
    for distortion in all_scores:
        scales = [s for s in all_scores[distortion].keys() if s is not None]
        if len(scales) <= 1:
            continue  # skip non-scaled types

        scales = sorted(scales)
        n_classes = len(centroids)
        n_cols = min(4, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        for idx, cls in enumerate(centroids):
            ax = axes[idx // n_cols, idx % n_cols]
            cls_scores = [all_scores[distortion][scale][cls] for scale in scales]
            ax.plot(scales, cls_scores, marker='o', label=f"{ALL_CLASSES[cls]}")
            # Plot original as a dashed line
            if orig_scores is not None:
                ax.axhline(orig_scores[cls], linestyle='--', color='red', alpha=0.9, label="Original", linewidth=2.5)
            # # Plot reversed as a dotted line if present
            # if reversed_scores is not None and distortion != "reversed":
            #     ax.axhline(reversed_scores[cls], linestyle=':', color='purple', alpha=0.7, label="Reversed")
            ax.set_title(f"{ALL_CLASSES[cls]}")
            ax.set_xlabel('Distortion Scale')
            ax.set_ylabel('Consistency Score')
            ax.set_ylim(0, 1.05)
            ax.legend(loc='best')
            ax.grid(True)

        # Remove empty axes if any
        for i in range(n_classes, n_rows * n_cols):
            fig.delaxes(axes[i // n_cols, i % n_cols])

        plt.suptitle(f"Consistency Score Trends per Class: {distortion.capitalize()}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(plot_dir, f"{distortion}_consistency_trends.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

    # ---- Plot "reversed" separately (one point per class, with "original" as reference) ----
    if "reversed" in all_scores and all_scores["reversed"].get(None, None) is not None:
        reversed_scores = all_scores["reversed"][None]
        n_classes = len(centroids)
        n_cols = min(4, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        for idx, cls in enumerate(centroids):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.bar([0], [reversed_scores[cls]], width=0.4, label="Reversed", color="purple")
            if orig_scores is not None:
                ax.axhline(orig_scores[cls], linestyle='--', color='red', alpha=0.9, label="Original", linewidth=2.5)
            ax.set_xticks([0])
            ax.set_xticklabels(['Reversed'])
            ax.set_ylabel('Consistency Score')
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{ALL_CLASSES[cls]}")
            ax.legend(loc='best')
            ax.grid(True, axis='y')

        # Remove empty axes if any
        for i in range(n_classes, n_rows * n_cols):
            fig.delaxes(axes[i // n_cols, i % n_cols])

        plt.suptitle(f"Consistency Score per Class: Reversed", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(plot_dir, f"reversed_consistency_per_class.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

    print("All plots saved to:", plot_dir)

def repeat_and_rewindow(seq, repeat_times=2, window_size=32):
    # seq: [T, D]
    rep_seq = seq.repeat_interleave(repeat_times, dim=0)  # [T*repeat_times, D]
    total_len = rep_seq.shape[0]
    num_windows = total_len // window_size
    windows = [rep_seq[i*window_size:(i+1)*window_size] for i in range(num_windows)]
    return windows

def test_embedding_sensitivity(
    model, test_loader, centroids, ALL_CLASSES, DEVICE,
    shuffle_fractions=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
    downsample_intervals=[2, 3, 4, 5],
    repetition_times=[2, 3, 4]
):
    import copy
    import numpy as np
    import torch

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

        # Shuffled frames: multiple fractions
        for frac in shuffle_fractions:
            shuf_embeds, shuf_labels = [], []
            for seqs, lengths, labels, vid_ids in test_loader:
                shuf_seqs = seqs.clone()
                shuf_seqs = partial_shuffle_within_window(shuf_seqs, lengths, vid_ids, shuffle_fraction=frac)
                shuf_seqs, lengths = shuf_seqs.to(DEVICE), lengths.to(DEVICE)
                emb, _ = model(shuf_seqs, lengths)
                shuf_embeds.append(emb.cpu())
                shuf_labels.append(labels)
            shuf_embeds = torch.cat(shuf_embeds)
            shuf_labels = torch.cat(shuf_labels)
            key = f"shuffled_{frac}"
            results[key] = (shuf_embeds, shuf_labels)

            diff = (base_embeds - shuf_embeds).norm(dim=1)
            print(f"Mean difference between original and shuffled (frac={frac}):", diff.mean().item())

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

    # Frame Downsample: multiple intervals
    for interval in downsample_intervals:
        drop_embeds, drop_labels = [], []
        for seqs, lengths, labels, vid_ids in test_loader:
            drop_seqs, new_lengths = frame_downsample(seqs, lengths, interval=interval)
            drop_seqs = torch.stack(drop_seqs, dim=0)  # Convert list to tensor
            drop_seqs, new_lengths = drop_seqs.to(DEVICE), new_lengths.to(DEVICE)
            emb, _ = model(drop_seqs, new_lengths)
            drop_embeds.append(emb.cpu())
            drop_labels.append(labels)
        drop_embeds = torch.cat(drop_embeds)
        drop_labels = torch.cat(drop_labels)
        key = f"dropout_{interval}"
        results[key] = (drop_embeds, drop_labels)
        diff_drop = (base_embeds - drop_embeds).norm(dim=1)
        print(f"Mean difference between original and dropout (interval={interval}):", diff_drop.mean().item())

    # Frame Repetition: multiple repeat_times
    for repeat in repetition_times:
        rep_windows = []
        rep_labels = []
        for seqs, lengths, labels, vid_ids in test_loader:
            for seq, l, label in zip(seqs, lengths, labels):
                seq = seq[:l]  # Remove padding
                # Repeat and re-window
                new_windows = repeat_and_rewindow(seq, repeat_times=repeat, window_size=32)
                rep_windows.extend(new_windows)
                rep_labels.extend([label] * len(new_windows))

        if len(rep_windows) == 0:
            continue  # nothing to process for this repeat

        # Batch for model
        rep_embeds = []
        batch_size = 256
        for i in range(0, len(rep_windows), batch_size):
            batch = torch.stack(rep_windows[i:i+batch_size]).to(DEVICE)  # [B, 32, D]
            emb, _ = model(batch)
            rep_embeds.append(emb.cpu())
        rep_embeds = torch.cat(rep_embeds)
        rep_labels = torch.tensor(rep_labels)
        key = f"repetition_{repeat}"
        results[key] = (rep_embeds, rep_labels)
        print(f"Processed repetition {repeat}, {len(rep_windows)} windows total.")
    # Consistency / centroid distance results
    for key in results:
        emb, labels = results[key]
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




def test():
    print(" Computing train embeddings...")

    model = TemporalTransformerV2Plus(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)
    print(f" Loading model from temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_NO_ENT.pt")
    model.load_state_dict(torch.load(f"SAVE/temporal_transformer_model_window_32_stride_8_valid_window_NO_ENT.pt"))
    model.eval()

    all_train_embeds = torch.load(f"SAVE/all_train_embeds_window_32_stride_8_valid_window_NO_ENT.pt")
    all_train_labels = torch.load(f"SAVE/all_train_labels_window_32_stride_8_valid_window_NO_ENT.pt")

    test_samples = torch.load(f"SAVE/test_samples_window_32_stride_8_valid_window.pt")
    test_labels = torch.load(f"SAVE/test_labels_window_32_stride_8_valid_window.pt")
    test_vid_ids = torch.load(f"SAVE/test_vid_ids_window_32_stride_8_valid_window.pt")

    class PoseVideoDatasetFromTensors(Dataset):
        def __init__(self, samples, labels, vid_ids):
            self.samples = torch.stack(samples)
            self.labels = torch.tensor(labels)
            self.vid_ids = vid_ids

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx], self.vid_ids[idx]

    # Create test dataset
    test_dataset = PoseVideoDatasetFromTensors(test_samples, test_labels, test_vid_ids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        # collate_fn=collate_fn
    )

    

    # Compute class centroids
    centroids = {}
    for cls in torch.unique(all_train_labels):
        mask = all_train_labels == cls
        centroid = all_train_embeds[mask].mean(dim=0)
        centroids[int(cls.item())] = centroid

    print(" Evaluating on test set...")
    model.eval()

    # Collect distances per (true_class, other_class)
    per_class_same_dists = {cls: [] for cls in centroids}
    per_class_to_other_dists = {cls: {other: [] for other in centroids if other != cls} for cls in centroids}

    with torch.no_grad():
        for seqs, labels, vid_ids in tqdm(test_loader):
            seqs = seqs.to(DEVICE)
            lengths = torch.full((seqs.shape[0],), seqs.shape[1], dtype=torch.long, device=DEVICE)
            emb, _, _ = model(seqs, lengths)
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
    print(f" Saved as embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    results = test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE)
    collect_and_plot_consistency_scores(results, centroids, ALL_CLASSES, plot_dir="SAVE_NEW2/consistency_plots")

    patch_transformer_encoder_layers(model)

    # def plot_cls_from_attention_mass(model, test_loader, window_size=32):
    #     model.eval()
    #     attn_from_cls_list = []
    #     for seqs, lengths, labels, vid_ids in test_loader:
    #         with torch.no_grad():
    #             _ = model(seqs.to(next(model.parameters()).device))
    #             attn_weights_per_layer =  [getattr(layer, "_saved_attn_weights", None) for layer in model.transformer.layers]
    #             # Each: [B, num_heads, T+1, T+1]
    #             # Get FROM CLS (first row) to all others (columns)
    #             attn_from_cls = torch.stack([aw[..., 0, 1:] for aw in attn_weights_per_layer])  # [layers, B, num_heads, T]
    #             attn_from_cls_list.append(attn_from_cls)
    #         break  # Just one batch for visualization
    #     attn_from_cls_all = torch.cat(attn_from_cls_list, dim=1)  # [layers, total_B, num_heads, T]
    #     avg_mass = attn_from_cls_all.mean(dim=(0,1,2)).numpy()
    #     plt.figure(figsize=(7,4))
    #     plt.plot(range(1, window_size+1), avg_mass[:window_size], marker="o")
    #     plt.title("Avg Attention FROM CLS Token to Frame Positions")
    #     plt.xlabel("Frame Position")
    #     plt.ylabel("Attention Mass")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig(f"SAVE_NEW2/attention_mass_cls_to_frames_window_{window_size}.png", dpi=200)
    #     print(f" Saved attention mass plot as attention_mass_cls_to_frames_window_{window_size}.png")
    def plot_cls_from_attention_mass(model, test_loader, window_size=32, num_batches=10):
        model.eval()
        attn_from_cls_list = []
        batch_counter = 0
        for seqs, lengths, labels, vid_ids in test_loader:
            with torch.no_grad():
                _ = model(seqs.to(next(model.parameters()).device))
                attn_weights_per_layer = [
                    getattr(layer, "_saved_attn_weights", None) for layer in model.transformer.layers
                ]
                # Each: [B, num_heads, T+1, T+1]
                # Get FROM CLS (first row) to all others (columns)
                attn_from_cls = torch.stack([aw[..., 0, 1:] for aw in attn_weights_per_layer])  # [layers, B, num_heads, T]
                attn_from_cls_list.append(attn_from_cls)
            batch_counter += 1
            # if batch_counter >= num_batches:
            #     break  # Only process num_batches

        attn_from_cls_all = torch.cat(attn_from_cls_list, dim=1)  # [layers, total_B, num_heads, T]
        avg_mass = attn_from_cls_all.mean(dim=(0,1,2)).numpy()
        plt.figure(figsize=(7,4))
        plt.plot(range(1, window_size+1), avg_mass[:window_size], marker="o")
        plt.title("Avg Attention FROM CLS Token to Frame Positions")
        plt.xlabel("Frame Position")
        plt.ylabel("Attention Mass")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"SAVE_NEW2/attention_mass_cls_to_frames_window_{window_size}.png", dpi=200)
        plt.show()
        print(f" Saved attention mass plot as attention_mass_cls_to_frames_window_{window_size}.png")

    plot_cls_from_attention_mass(model, test_loader, window_size=32)

# ——— MAIN ———
if __name__ == "__main__":
    test()