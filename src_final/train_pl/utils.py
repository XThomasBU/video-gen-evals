import cv2
from torch.utils.data import Sampler, BatchSampler
from collections import defaultdict
import numpy as np
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
import torch.nn.functional as F
from dataclasses import dataclass
import sys
import json
import math
import random
import typing as T
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NGPUS = torch.cuda.device_count()
USE_DP = (DEVICE == "cuda") and (NGPUS > 1)
PRIMARY_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")
g = torch.Generator(); g.manual_seed(SEED)

def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def partial_shuffle_within_window(seqs, shuffle_fraction=0.7):
    shuffled = seqs.clone()
    batch_size, max_len, feat_dim = seqs.shape
    for i in range(batch_size):
        l = max_len
        if l > 1:
            n_to_shuffle = max(1, int(shuffle_fraction * l))
            indices = torch.randperm(l)[:n_to_shuffle]
            shuffled_part = shuffled[i, indices][torch.randperm(n_to_shuffle)]
            shuffled[i, indices] = shuffled_part
    return shuffled


def reverse_sequence(seqs):
    # [B, T, D] → reversed in T dim
    reversed_seqs = []
    batch_size, max_len, feat_dim = seqs.shape
    for i in range(batch_size):
        l = max_len
        reversed = torch.flip(seqs[i, :l], dims=[0])
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


from math import log


class SUPCON(nn.Module):
    def __init__(self, temperature=0.1):

        super(SUPCON, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):

        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T)
        exp_dot_tempered = torch.exp((dot_product_tempered) / self.temperature) 
       

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_positives = mask_similar_class * mask_anchor_out
        mask_negatives = ~mask_similar_class
        positives_per_samples = torch.sum(mask_positives, dim=1)
        negatives_per_samples = torch.sum(mask_negatives, dim=1)
        
        supcon_loss = torch.sum(-torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_positives, dim=1)+(torch.sum(exp_dot_tempered * mask_negatives, dim=1)))) * mask_positives,dim=1) / positives_per_samples
        
        supcon_loss_mean = torch.mean(supcon_loss)
        return supcon_loss_mean


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


class PKBatchSampler(BatchSampler):
    """
    Balanced class sampler for metric learning: each batch has P classes and K samples per class.
    Batch size = P * K.
    If a class has < K items remaining in its epoch-queue, samples with replacement to fill.
    """
    def __init__(self, labels, P, K, drop_last=False, generator=None):
        """
        labels: sequence of ints (len = N)
        P: number of classes per batch
        K: samples per class (min positives per class)
        """
        self.labels = np.asarray(labels)
        self.P = int(P)
        self.K = int(K)
        self.drop_last = drop_last
        self.rng = np.random.default_rng() if generator is None else generator

        # Build class -> indices mapping
        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[int(y)].append(idx)
        self.classes = list(self.class_to_indices.keys())

        assert len(self.classes) >= self.P, "P must be <= number of classes"

        # Precompute epoch state
        self._reset_epoch()

    def _reset_epoch(self):
        # For each class, make a shuffled queue of its indices
        self.per_class_queues = {}
        for c, idxs in self.class_to_indices.items():
            idxs = np.array(idxs)
            self.rng.shuffle(idxs)
            self.per_class_queues[c] = idxs.tolist()
        # Shuffle class order for diversity
        self.class_order = self.classes.copy()
        self.rng.shuffle(self.class_order)
        self.class_cursor = 0

        # Estimate number of batches in epoch (roughly balances coverage)
        total_items = sum(len(v) for v in self.per_class_queues.values())
        self.num_batches = total_items // (self.P * self.K)

    def __iter__(self):
        self._reset_epoch()
        batches_emitted = 0

        while True:
            # pick P classes for this batch by cycling through shuffled class list
            if self.class_cursor + self.P <= len(self.class_order):
                chosen = self.class_order[self.class_cursor : self.class_cursor + self.P]
                self.class_cursor += self.P
            else:
                # wrap and reshuffle for the next cycle
                remaining = len(self.class_order) - self.class_cursor
                chosen = self.class_order[self.class_cursor:] + self.class_order[:self.P - remaining]
                self.rng.shuffle(self.class_order)
                self.class_cursor = self.P - remaining

            batch = []
            for c in chosen:
                q = self.per_class_queues[c]
                if len(q) >= self.K:
                    # take K without replacement
                    take = q[:self.K]
                    del q[:self.K]
                else:
                    # take what’s left and top up with replacement
                    take = q.copy()
                    need = self.K - len(take)
                    pool = self.class_to_indices[c]
                    fill = self.rng.choice(pool, size=need, replace=True).tolist()
                    take.extend(fill)
                    q.clear()
                batch.extend(take)

            # optional per-batch shuffle
            self.rng.shuffle(batch)

            if len(batch) != self.P * self.K:
                if self.drop_last:
                    continue
            yield batch
            batches_emitted += 1

            if batches_emitted >= self.num_batches:
                break

    def __len__(self):
        # approximate number of batches per epoch
        total_items = sum(len(v) for v in self.class_to_indices.values())
        return total_items // (self.P * self.K)


def get_static_window(seqs):
    # static window -- replace window with the first frame of the sequence
    static_seqs = []
    for seq in seqs:
        first_frame = seq[0].unsqueeze(0)  # [1, D]
        static_seq = first_frame.repeat(seq.shape[0], 1)  # [T, D]
        static_seqs.append(static_seq)
    return torch.stack(static_seqs, dim=0)  # [B, T, D]


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
        for seqs, lengths, labels, vid_ids, window_ids in test_loader:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _ = model(seqs, lengths)
            base_embeds.append(emb.cpu())
            base_labels.append(labels)
        base_embeds = torch.cat(base_embeds)
        base_labels = torch.cat(base_labels)
        results["original"] = (base_embeds, base_labels)

        # Shuffled frames
        shuf_embeds, shuf_labels = [], []
        for seqs, lengths, labels, vid_ids, window_ids in test_loader:
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
        for seqs, lengths, labels, vid_ids, window_ids in test_loader:
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
    for seqs, lengths, labels, vid_ids, window_ids in test_loader:
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
    for seqs, lengths, labels, vid_ids, window_ids in test_loader:
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
    for seqs, lengths, labels, vid_ids, window_ids in test_loader:
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

def get_next_window_index(vid_ids, window_ids, current_vid, current_window):
    # Find all indices for the given video
    vid_indices = [i for i, vid in enumerate(vid_ids) if vid == current_vid]
    # Find window_ids for this video and sort
    win_for_vid = [(window_ids[i], i) for i in vid_indices]
    win_for_vid.sort()  # sort by window index

    # Find the position of the current window
    pos = [i for i, (w, _) in enumerate(win_for_vid) if w == current_window]
    if not pos:
        return None
    pos = pos[0]
    # If there is a next window, return its index
    if pos + 1 < len(win_for_vid):
        return win_for_vid[pos + 1][1]  # index in original arrays
    else:
        return None  # No next window


def collate_fn(batch):
    sequences, labels, vid_ids, window_ids = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)  # [B, T_max, D]
    labels = torch.tensor(labels)
    return sequences, lengths, labels, vid_ids, window_ids


def plot_loss_plot(loss_dict, WINDOW_SIZE, STRIDE):
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
    plt.savefig(f"SAVE_PLOTS/loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")
    print(f" Saved as loss_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")

def plot_weights(weights_dict):
    # plot weights
    fig, axs = plt.subplots(5, figsize=(20, 12))
    for i, (key, values) in enumerate(weights_dict.items()):
        axs[i].plot(values, label=key, color=plt.cm.tab10(i), linewidth=2.5)
        axs[i].set_title(f'{key.capitalize()} Weights')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"SAVE_PLOTS/weights_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")
    print(f" Saved as weights_curves_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")


def post_training(loss_dict, model, train_loader, test_loader, ALL_CLASSES, WINDOW_SIZE, STRIDE, DEVICE):

    print(" Saving loss curves...")
    plot_loss_plot(loss_dict, WINDOW_SIZE, STRIDE)

    print(" Computing train embeddings...")
    model.eval()
    all_train_embeds = []
    all_train_labels = []
    all_train_vid_ids = []   # <-- NEW
    with torch.no_grad():
        for seqs, lengths, labels, vid_ids, window_ids in tqdm(train_loader):
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            emb, _, _ = model(seqs, lengths)
            all_train_embeds.append(emb.cpu())
            all_train_labels.append(labels.cpu())
            all_train_vid_ids.extend(vid_ids)  # vid_ids should be a list/tuple of strings or ints
    all_train_embeds = torch.cat(all_train_embeds)
    all_train_labels = torch.cat(all_train_labels)
    all_train_vid_ids = np.array(all_train_vid_ids)

    # save all_train_embeds and all_train_labels
    torch.save(all_train_embeds, f"SAVE/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    torch.save(all_train_labels, f"SAVE/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    torch.save(all_train_vid_ids, f"SAVE/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")

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

    with open(f"SAVE/centroids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pkl", "wb") as f:
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
            emb, _, _ = model(seqs, lengths)
            emb = emb.cpu()
            all_test_embeds.append(emb)
            all_test_labels.append(labels)
    all_test_embeds = torch.cat(all_test_embeds)
    all_test_labels = torch.cat(all_test_labels)


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
    colors = plt.cm.get_cmap("tab20", len(ALL_CLASSES))

    plt.figure(figsize=(10, 8))

    # Plot train embeddings
    for cls in range(len(ALL_CLASSES)):
        mask = (all_train_labels == cls).numpy()
        plt.scatter(
            projected_train_embeds[mask, 0],
            projected_train_embeds[mask, 1],
            s=17,
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
    plt.savefig(f"SAVE_PLOTS/embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png", dpi=200)
    print(f" Saved as embeddings_centroids_with_test_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")
    plt.close()

def load_all_frames(video_path, convert_bgr2rgb):
    """Load every frame from a video into memory (no subsampling, no cap)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if convert_bgr2rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    return frames

# ============================== SO(3) utils ==============================

def _axis_angle_to_matrix(a: torch.Tensor) -> torch.Tensor:
    """Axis-angle -> rotation matrix via Rodrigues. a: [...,3] -> [...,3,3]"""
    theta = a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = a / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    O = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([O,   -kz,  ky], dim=-1),
        torch.stack([kz,    O, -kx], dim=-1),
        torch.stack([-ky,  kx,   O], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(a.shape[:-1] + (3, 3))
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return I + s * K + (1.0 - c) * (K @ K)

def _log_so3(R: torch.Tensor) -> torch.Tensor:
    """Matrix log on SO(3) -> axis-angle vector. R: [...,3,3] -> [...,3]"""
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1 + 1e-6, 3 - 1e-6)
    theta = torch.acos((tr - 1) / 2)
    denom = (2 * torch.sin(theta)).unsqueeze(-1).clamp_min(1e-6)
    v = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) / denom
    return theta.unsqueeze(-1) * v

def _vit_delta(vit: torch.Tensor) -> torch.Tensor:
    """Cosine-stable feature change. vit: [T,D] -> [T,D]"""
    # if numpy array, convert to tensor
    v = F.normalize(vit, dim=-1)
    v_prev = torch.cat([v[:1], v[:-1]], dim=0)
    return v - v_prev

def _rot_axisangle_delta(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle pose -> SO(3) relative delta via log map. aa: [T,3*J] -> [T,3*J]"""
    T, D = aa.shape
    J = D // 3
    a = aa.view(T, J, 3)
    a_prev = torch.cat([a[:1], a[:-1]], dim=0)
    R = _axis_angle_to_matrix(a)
    R0 = _axis_angle_to_matrix(a_prev)
    Rrel = torch.matmul(R0.transpose(-1, -2), R)
    w = _log_so3(Rrel)
    return w.view(T, D)

def _betas_delta(betas: torch.Tensor) -> torch.Tensor:
    diff = betas - torch.cat([betas[:1], betas[:-1]], dim=0)
    return diff

# ============================== Data layer ==============================

@dataclass
class VideoItem:
    cls: str
    name: str   # file name with .npz
    path: str
    length: int # number of frames (T)
    vit_dim: int

class NpzVideoDataset(Dataset):
    """
    Scans per-class directories for .npz files saved by save_video_npz(...).
    """
    def __init__(self, root_dir: str, items: T.Optional[T.List[VideoItem]] = None,
                 whitelist_json_dir: T.Optional[str] = None, filter_classes: T.Optional[T.List[str]] = None, min_videos_per_class: int = 10, enforce_min_per_class: bool = True):
        self.root_dir = root_dir
        self.whitelist = self._load_whitelist(whitelist_json_dir) if whitelist_json_dir else {}

        raw_items = items if items is not None else self._scan()

        # group items by class
        class_to_items: T.Dict[str, T.List[VideoItem]] = {}
        for it in raw_items:
            class_to_items.setdefault(it.cls, []).append(it)

        if enforce_min_per_class:
            class_to_items = {
                cls: vids for cls, vids in class_to_items.items()
                if len(vids) >= min_videos_per_class
            }

        # optional explicit class filter
        if filter_classes is not None:
            allowed = set(filter_classes)
            class_to_items = {cls: vids for cls, vids in class_to_items.items() if cls in allowed}

        # flatten
        self.class_to_items = class_to_items
        self.items = [it for vids in class_to_items.values() for it in vids]
        self.classes = sorted(class_to_items.keys())

    def _load_whitelist(self, wdir: str) -> T.Dict[str, T.Set[str]]:
        wl: T.Dict[str, T.Set[str]] = {}
        if os.path.isdir(wdir):
            for fname in sorted(os.listdir(wdir)):
                if fname.endswith(".json"):
                    cls_name = os.path.splitext(fname)[0]
                    with open(os.path.join(wdir, fname), "r") as f:
                        vids = json.load(f)
                    # whitelist files may store original video names; we accept either stem or stem+".npz"
                    wl[cls_name] = set(os.path.splitext(os.path.basename(v))[0] for v in vids)
        return wl

    def _scan(self) -> T.List[VideoItem]:
        items: T.List[VideoItem] = []
        for cls in sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]):
            cls_dir = os.path.join(self.root_dir, cls)
            for f in sorted(os.listdir(cls_dir)):
                if not f.endswith(".npz"):
                    continue
                stem = os.path.splitext(f)[0]
                if self.whitelist and (stem not in self.whitelist.get(cls, set())):
                    continue
                path = os.path.join(cls_dir, f)
                try:
                    npz = np.load(path, mmap_mode="r")
                    # primary arrays
                    pose = npz["pose"]       # [T,69]
                    vit  = npz["vit"]        # [T,D]
                    Tlen = pose.shape[0]
                    Dvit = vit.shape[1]
                    items.append(VideoItem(cls=cls, name=f, path=path, length=Tlen, vit_dim=Dvit))
                except Exception:
                    # Skip corrupted entries silently; you can log if desired
                    continue
        return items

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def train_test_split(dataset: NpzVideoDataset, train_ratio: float = 0.8, seed: int = 42):
    rng = random.Random(seed)
    train_items: T.List[VideoItem] = []
    test_items:  T.List[VideoItem] = []

    for cls, vids in dataset.class_to_items.items():
        vids_copy = vids[:]
        rng.shuffle(vids_copy)
        n = len(vids_copy)
        n_train = max(1, min(n - 1, int(round(n * train_ratio))))  # ensure both sides non-empty
        train_items.extend(vids_copy[:n_train])
        test_items.extend(vids_copy[n_train:])

    train_ds = NpzVideoDataset(dataset.root_dir, items=train_items, enforce_min_per_class=False)
    test_ds  = NpzVideoDataset(dataset.root_dir, items=test_items, enforce_min_per_class=False)
    return train_ds, test_ds

# ------------------- window sampling from NPZ -------------------

def sample_windows_capped_npz(
    ds: NpzVideoDataset,
    clip_len: int = 32,
    stride:   int = 1,
    windows_per_video: int = 4,
    total_cap: int = 1000,
    seed: int = 1337
):
    """
    Returns up to total_cap tuples: (VideoItem, start)
    """
    rng = random.Random(seed)
    vids = ds.items[:]
    rng.shuffle(vids)

    out: T.List[T.Tuple[VideoItem, int]] = []
    for it in vids:
        if len(out) >= total_cap:
            break
        max_start = max(0, it.length - clip_len)
        if max_start <= 0:
            continue
        possible = list(range(0, max_start + 1, stride))
        k = min(windows_per_video, len(possible), total_cap - len(out))
        starts = rng.sample(possible, k)
        for s in starts:
            out.append((it, s))
            if len(out) >= total_cap:
                break
    rng.shuffle(out)
    return out

# ----------------------- Window dataset ------------------------

class WindowDataset(Dataset):
    """
    Loads windows from .npz arrays and constructs features + motion deltas.
    """
    def __init__(self, samples: T.List[T.Tuple[VideoItem,int]],
                 clip_len: int = 32,
                 retries: int = 2,
                 jitter: int = 16,
                 pad_mode: str = "repeat",
                 seed: int = 1337,
                 stats: T.Optional['ModalityStats'] = None):
        self.samples = samples
        self.clip_len = clip_len
        self.retries = retries
        self.jitter = jitter
        self.pad_mode = pad_mode
        self.rng = random.Random(seed)
        self.stats = stats

    def __len__(self): return len(self.samples)

    def _slice_or_pad(self, arr: np.ndarray, start: int, T: int) -> np.ndarray:
        """
        arr: [N, ...], take arr[start:start+T]; if short, pad by nearest-repeat.
        """
        end = start + T
        if start < 0 or start >= arr.shape[0]:
            # fallback to repeating first or last frame
            idx = 0 if start < 0 else arr.shape[0] - 1
            return np.repeat(arr[idx:idx+1], T, axis=0)
        if end <= arr.shape[0]:
            return arr[start:end]
        # tail short -> pad with last available frame
        tail = arr[start:]
        need = T - tail.shape[0]
        pad = np.repeat(arr[-1:], need, axis=0)
        return np.concatenate([tail, pad], axis=0)

    def _try_one(self, it: VideoItem, start: int):
        npz = np.load(it.path, mmap_mode="r")  # zero-copy reads
        pose = npz["pose"]            # [N, J=23, 3, 3] rotation matrices
        betas = npz["betas"]          # [N, 10]
        gori = npz["global_orient"]   # [N, 3, 3] rotation matrices
        vit = npz["vit"]              # [N, D]  (expected D == dims_map['vit'], e.g., 1024)

        # slice (or repeat-pad at edges)
        pose_w  = self._slice_or_pad(pose,  start, self.clip_len)  # [T,J,3,3]
        betas_w = self._slice_or_pad(betas, start, self.clip_len)  # [T,10]
        gori_w  = self._slice_or_pad(gori,  start, self.clip_len)  # [T,3,3]
        vit_w   = self._slice_or_pad(vit,   start, self.clip_len)  # [T,D]
        T = pose_w.shape[0]

        # tensors
        pose_R = torch.from_numpy(pose_w).float()   # [T,J,3,3]
        gori_R = torch.from_numpy(gori_w).float()   # [T,3,3]
        betas_t = torch.from_numpy(betas_w).float() # [T,10]
        vit_t   = torch.from_numpy(vit_w).float()   # [T,D]

        # ---------- RAW (state) ----------
        pose_raw  = pose_R.reshape(T, -1)           # [T, 9*J] = 207
        gori_raw  = gori_R.reshape(T, -1)           # [T, 9]
        vit_raw   = vit_t                           # [T, D]
        beta_raw  = betas_t                         # [T, 10]

        # ---------- MOTION (diff) ----------
        vit_diff = _vit_delta(vit_t)                 # [T, D]
        gori_diff = _rot_axisangle_delta(gori_raw)  # [T, 3]
        pose_diff = _rot_axisangle_delta(pose_raw)  # [T, 9*J]
        beta_diff = _betas_delta(beta_raw)          # [T, 10]


        if self.stats is not None:
            eps = 1e-6
            vit_raw  = (vit_raw  - self.stats.vit_raw_mean)  / (self.stats.vit_raw_std  + eps)
            gori_raw = (gori_raw - self.stats.gori_raw_mean) / (self.stats.gori_raw_std + eps)
            pose_raw = (pose_raw - self.stats.pose_raw_mean) / (self.stats.pose_raw_std + eps)
            beta_raw = (beta_raw - self.stats.beta_raw_mean) / (self.stats.beta_raw_std + eps)

            vit_diff  = (vit_diff  - self.stats.vit_diff_mean)  / (self.stats.vit_diff_std  + eps)
            gori_diff = (gori_diff - self.stats.gori_diff_mean) / (self.stats.gori_diff_std + eps)
            pose_diff = (pose_diff - self.stats.pose_diff_mean) / (self.stats.pose_diff_std + eps)
            beta_diff = (beta_diff - self.stats.beta_diff_mean) / (self.stats.beta_diff_std + eps)

        raw  = torch.cat([vit_raw,  gori_raw,  pose_raw,  beta_raw], dim=-1)
        diff = torch.cat([vit_diff, gori_diff, pose_diff, beta_diff], dim=-1)
        feats = torch.cat([raw, diff], dim=-1)

        return feats, it.cls, it.name

    def __getitem__(self, idx):
        it, start = self.samples[idx]
        out = self._try_one(it, start)
        if out is not None:
            return out
        # retry near start with jitter
        for _ in range(self.retries):
            delta = self.rng.randint(-self.jitter, self.jitter)
            ns = max(0, min(start + delta, max(0, it.length - self.clip_len)))
            out = self._try_one(it, ns)
            if out is not None:
                return out
        return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    feats, cls_names, vids = zip(*batch)
    feats = torch.stack(feats, dim=0)
    return feats, list(cls_names), list(vids)

from dataclasses import dataclass

@dataclass
class ModalityStats:
    vit_raw_mean:  torch.Tensor; vit_raw_std:  torch.Tensor
    gori_raw_mean: torch.Tensor; gori_raw_std: torch.Tensor
    pose_raw_mean: torch.Tensor; pose_raw_std: torch.Tensor
    beta_raw_mean: torch.Tensor; beta_raw_std: torch.Tensor

    vit_diff_mean:  torch.Tensor; vit_diff_std:  torch.Tensor
    gori_diff_mean: torch.Tensor; gori_diff_std: torch.Tensor
    pose_diff_mean: torch.Tensor; pose_diff_std: torch.Tensor
    beta_diff_mean: torch.Tensor; beta_diff_std: torch.Tensor

import numpy as np
import torch

def _update_sum_sum2(X: np.ndarray, sum1: np.ndarray, sum2: np.ndarray):
    # X: [T, D]
    sum1 += X.sum(axis=0, dtype=np.float64)
    sum2 += (X.astype(np.float64) ** 2).sum(axis=0)
    return sum1, sum2, X.shape[0]

def compute_stats_from_npz(train_items: T.List[VideoItem], eps: float = 1e-6) -> ModalityStats:
    """
    Stream over TRAIN .npz files and compute per-dim mean/std for:
      vit_raw, gori_raw, pose_raw, beta_raw, and their DIFF counterparts.
    Uses exactly the same diff definitions as your dataset code:
      - vit_diff: L2-normalize per-frame then v - v_prev (first row self-diff)
      - gori_diff / pose_diff: (R_prev^T R_cur - I).reshape(...)
      - beta_diff: finite difference with clamp [-0.2, 0.2]
    """
    # Infer dims from first file
    assert len(train_items) > 0
    npz0 = np.load(train_items[0].path, mmap_mode="r")
    vitD = int(npz0["vit"].shape[1])
    J    = int(npz0["pose"].shape[1])         # 23
    D_pose = 9 * J
    D_gori = 9
    D_beta = 10
    D_vit  = vitD

    # Allocate accumulators (float64 for numerical stability)
    def zeros(D): return np.zeros((D,), dtype=np.float64)

    s_vit_raw  = zeros(D_vit);  ss_vit_raw  = zeros(D_vit);  n_vit_raw  = 0
    s_gori_raw = zeros(D_gori); ss_gori_raw = zeros(D_gori); n_gori_raw = 0
    s_pose_raw = zeros(D_pose); ss_pose_raw = zeros(D_pose); n_pose_raw = 0
    s_beta_raw = zeros(D_beta); ss_beta_raw = zeros(D_beta); n_beta_raw = 0

    s_vit_diff  = zeros(D_vit);  ss_vit_diff  = zeros(D_vit);  n_vit_diff  = 0
    s_gori_diff = zeros(D_gori); ss_gori_diff = zeros(D_gori); n_gori_diff = 0
    s_pose_diff = zeros(D_pose); ss_pose_diff = zeros(D_pose); n_pose_diff = 0
    s_beta_diff = zeros(D_beta); ss_beta_diff = zeros(D_beta); n_beta_diff = 0

    I3 = np.eye(3, dtype=np.float32)

    for it in train_items:
        npz = np.load(it.path, mmap_mode="r")
        pose = npz["pose"].astype(np.float32)        # [T,J,3,3]
        gori = npz["global_orient"].astype(np.float32) # [T,3,3]
        betas = npz["betas"].astype(np.float32)      # [T,10]
        vit = npz["vit"].astype(np.float32)          # [T,D]
        Tlen = pose.shape[0]

        # ---------- RAW ----------
        pose_raw  = pose.reshape(Tlen, -1)           # [T, 9*J]
        gori_raw  = gori.reshape(Tlen, -1)           # [T, 9]
        vit_raw   = vit                               # [T, D]
        beta_raw  = betas                             # [T, 10]

        s_vit_raw,  ss_vit_raw,  c = _update_sum_sum2(vit_raw,  s_vit_raw,  ss_vit_raw);  n_vit_raw  += c
        s_gori_raw, ss_gori_raw, c = _update_sum_sum2(gori_raw, s_gori_raw, ss_gori_raw); n_gori_raw += c
        s_pose_raw, ss_pose_raw, c = _update_sum_sum2(pose_raw, s_pose_raw, ss_pose_raw); n_pose_raw += c
        s_beta_raw, ss_beta_raw, c = _update_sum_sum2(beta_raw, s_beta_raw, ss_beta_raw); n_beta_raw += c

        # ---------- DIFF ----------
        # vit: normalize per frame, then diff to previous (first row self-diff)
        vit_diff = _vit_delta(torch.from_numpy(vit_raw).float()).numpy()  # [T, D]
        gori_diff = _rot_axisangle_delta(torch.from_numpy(gori_raw).float()).numpy()  # [T, 3]
        pose_diff = _rot_axisangle_delta(torch.from_numpy(pose_raw).float()).numpy()  # [T, 9*J]
        beta_diff = _betas_delta(torch.from_numpy(beta_raw).float()).numpy()                # [T, 10]
        # vit_diff, gori_diff, pose_diff, beta_diff - vit_diff.numpy(), gori_diff.numpy(), pose_diff.numpy(), beta_diff.numpy()

        s_vit_diff,  ss_vit_diff,  c = _update_sum_sum2(vit_diff,  s_vit_diff,  ss_vit_diff);  n_vit_diff  += c
        s_gori_diff, ss_gori_diff, c = _update_sum_sum2(gori_diff, s_gori_diff, ss_gori_diff); n_gori_diff += c
        s_pose_diff, ss_pose_diff, c = _update_sum_sum2(pose_diff, s_pose_diff, ss_pose_diff); n_pose_diff += c
        s_beta_diff, ss_beta_diff, c = _update_sum_sum2(beta_diff, s_beta_diff, ss_beta_diff); n_beta_diff += c

    # finalize mean/std (per-dim)
    def finalize(sum1, sum2, n):
        mean = sum1 / max(1, n)
        var  = sum2 / max(1, n) - mean**2
        std  = np.sqrt(np.maximum(var, 0.0) + eps)
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))

    vit_raw_mean,  vit_raw_std  = finalize(s_vit_raw,  ss_vit_raw,  n_vit_raw)
    gori_raw_mean, gori_raw_std = finalize(s_gori_raw, ss_gori_raw, n_gori_raw)
    pose_raw_mean, pose_raw_std = finalize(s_pose_raw, ss_pose_raw, n_pose_raw)
    beta_raw_mean, beta_raw_std = finalize(s_beta_raw, ss_beta_raw, n_beta_raw)

    vit_diff_mean,  vit_diff_std  = finalize(s_vit_diff,  ss_vit_diff,  n_vit_diff)
    gori_diff_mean, gori_diff_std = finalize(s_gori_diff, ss_gori_diff, n_gori_diff)
    pose_diff_mean, pose_diff_std = finalize(s_pose_diff, ss_pose_diff, n_pose_diff)
    beta_diff_mean, beta_diff_std = finalize(s_beta_diff, ss_beta_diff, n_beta_diff)

    return ModalityStats(
        vit_raw_mean, vit_raw_std,
        gori_raw_mean, gori_raw_std,
        pose_raw_mean, pose_raw_std,
        beta_raw_mean, beta_raw_std,
        vit_diff_mean, vit_diff_std,
        gori_diff_mean, gori_diff_std,
        pose_diff_mean, pose_diff_std,
        beta_diff_mean, beta_diff_std,
    )


def make_test_loader(
    ds: NpzVideoDataset,
    clip_len: int,
    stride: int,
    pad_mode: str = "repeat",
    stats: T.Optional['ModalityStats'] = None,
    seed: int = 999,
    batch_size: int = 256,
    num_workers: int = 0,
    filter_classes: T.Optional[T.List[str]] = None,
):
    """
    Enumerate *all* windows for every video in `ds` with the given stride.
    No sampling, no caps. If a video is shorter than clip_len, you'll get one
    padded window (start=0).
    """
    allowed: T.Optional[set] = set(filter_classes) if filter_classes else None

    samples: T.List[T.Tuple[VideoItem, int]] = []
    for it in ds.items:
        if allowed is not None and it.cls not in allowed:
            continue
        if it.length <= 0:
            continue

        if it.length < clip_len:
            starts = [0]  # one padded window
        else:
            last_start = it.length - clip_len
            starts = list(range(0, last_start + 1, max(1, stride)))

        for s in starts:
            samples.append((it, s))

    # (Optional) guardrail: avoid silent empty loader
    if len(samples) == 0:
        raise ValueError(
            f"make_test_loader: no samples found. "
            f"{'Filter matched no classes.' if allowed else 'Dataset may be empty.'}"
        )

    ds_all = WindowDataset(
        samples,
        clip_len=clip_len,
        retries=0,
        jitter=0,
        pad_mode=pad_mode,
        seed=seed,
        stats=stats,
    )

    return DataLoader(
        ds_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda"),
    )
@torch.no_grad()
def action_consistency(
    model,
    loader,
    label_dict,
    device=PRIMARY_DEVICE,
    metric: str = "cosine",   # "cosine" | "l2" | "l2sq"
):
    model.eval()
    all_z, all_y, all_vid = [], [], []

    for packed in loader:
        if packed is None:
            continue
        feats, cls_names, vids = packed
        feats = feats.to(device, non_blocking=True)
        z, _, _ = model(feats)          # [B, D]
        # If metric == "cosine", embeddings should be unit norm.
        if metric == "cosine":
            z = F.normalize(z, dim=-1)

        y = torch.as_tensor([label_dict[c] for c in cls_names], device=z.device, dtype=torch.long)
        all_z.append(z)
        all_y.append(y)
        all_vid.extend(vids)            # keep Python list of video names

    if not all_z:
        return {}

    Z = torch.cat(all_z, dim=0)         # [N, D]
    Y = torch.cat(all_y, dim=0)         # [N]
    vids_tensor = np.array(all_vid)     # [N] (numpy for grouping)

    # Pairwise distances
    if metric == "cosine":
        S = Z @ Z.t()
        D = (1.0 - S)                    # [N, N]
    elif metric == "l2":
        D = torch.cdist(Z, Z, p=2)
    elif metric == "l2sq":
        D = torch.cdist(Z, Z, p=2) ** 2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    N = D.shape[0]
    eye = torch.eye(N, device=D.device, dtype=torch.bool)

    # Build quick lookups
    # Map class id per row
    Y_cpu = Y.detach().cpu().numpy()
    # Group indices per video name
    from collections import defaultdict
    inds_by_video: T.Dict[str, T.List[int]] = defaultdict(list)
    for i, vn in enumerate(all_vid):
        inds_by_video[vn].append(i)

    results = {}   # per-video stats
    for vn, idx_list in inds_by_video.items():
        idx = torch.as_tensor(idx_list, device=D.device, dtype=torch.long)
        if idx.numel() < 2:
            # one window only: intra is undefined (set to nan)
            intra_vals = torch.tensor([], device=D.device)
        else:
            Dc = D[idx][:, idx]
            # drop diagonal
            mask_intra = ~torch.eye(len(idx), device=D.device, dtype=torch.bool)
            intra_vals = Dc[mask_intra]

        # Inter = compare this video's windows to windows of *different classes*
        cls_v = Y[idx[0]].item()    # all windows of a video share the same class
        mask_diff_class = (Y != cls_v)
        inter_vals = D[idx][:, mask_diff_class].reshape(-1)

        intra_mean = (intra_vals.mean().item() if intra_vals.numel() > 0 else float("nan"))
        inter_mean = (inter_vals.mean().item() if inter_vals.numel() > 0 else float("nan"))

        if np.isfinite(intra_mean) and np.isfinite(inter_mean) and (inter_mean + intra_mean) > 0:
            score = inter_mean / (inter_mean + intra_mean)
        else:
            score = float("nan")

        results[vn] = {
            "class": int(cls_v),
            "windows": int(idx.numel()),
            "intra": intra_mean,
            "inter": inter_mean,
            "score": score,
        }

    avg_score = np.nanmean([v["score"] for v in results.values()]) if results else float("nan")
    max_score = np.nanmax([v["score"] for v in results.values()]) if results else float("nan")
    min_score = np.nanmin([v["score"] for v in results.values()]) if results else float("nan")
    median_score = np.nanmedian([v["score"] for v in results.values()]) if results else float("nan")
    classes_with_score_greater_than_0_9 = sum(1 for v in results.values() if np.isfinite(v["score"]) and v["score"] > 0.9)

    model.train()
    return results, avg_score, max_score, min_score, median_score


@torch.no_grad()
def action_consistency_with_centroids(
    model,
    loader,                  # test loader (uses stats=stats)
    label_dict,              # {class_name: class_id}
    centroids,               # [C, D], ideally unit-norm
    device=PRIMARY_DEVICE,
):
    model.eval()
    centroids = F.normalize(centroids, dim=-1)  # safety
    id2cls = {i: c for c, i in label_dict.items()}

    # buckets for window-level accum
    # we will aggregate to videos, then to classes
    from collections import defaultdict
    win_accum = defaultdict(lambda: {"intra_sum": 0.0, "inter_sum": 0.0, "n": 0, "cls_id": None})

    for packed in loader:
        if packed is None:
            continue
        feats, cls_names, vids = packed
        feats = feats.to(device, non_blocking=True)
        z, _, _ = model(feats)                  # [B, D]
        z = F.normalize(z, dim=-1)              # cosine geometry

        y = torch.as_tensor([label_dict[c] for c in cls_names], device=device, dtype=torch.long)  # [B]
        sims = z @ centroids.t()                # [B, C]
        d_all = 1.0 - sims                      # cosine distance to each centroid

        # gather intra and mean inter per window
        b_idx = torch.arange(y.numel(), device=device)
        d_intra = d_all[b_idx, y]               # [B]

        # mean over other classes
        C = centroids.size(0)
        mask = torch.ones_like(d_all, dtype=torch.bool)
        mask[b_idx, y] = False
        # sum distances to others, divide by (C-1)
        d_inter = (d_all.masked_fill(~mask, 0.0).sum(dim=1)) / max(1, C - 1)

        # accumulate per-video
        for vid, yi, din, dout in zip(vids, y.tolist(), d_intra.tolist(), d_inter.tolist()):
            a = win_accum[vid]
            a["intra_sum"] += float(din)
            a["inter_sum"] += float(dout)
            a["n"] += 1
            if a["cls_id"] is None:
                a["cls_id"] = yi

    # compute per-video scores
    video_scores = {}
    for vid, a in win_accum.items():
        n = max(1, a["n"])
        intra_mean = a["intra_sum"] / n
        inter_mean = a["inter_sum"] / n
        denom = inter_mean + intra_mean
        score = (inter_mean / denom) if denom > 0 else float("nan")
        video_scores[vid] = {
            "class": a["cls_id"],
            "windows": a["n"],
            "intra": intra_mean,
            "inter": inter_mean,
            "score": score,
        }

    # aggregate per-class from *video* scores
    per_class_scores = defaultdict(list)
    for vid, d in video_scores.items():
        per_class_scores[d["class"]].append(d["score"])

    class_stats = {
        id2cls[cls_id]: {
            "count_videos": len(scores),
            "avg": float(np.nanmean(scores)) if scores else float("nan"),
            "max": float(np.nanmax(scores))  if scores else float("nan"),
            "min": float(np.nanmin(scores))  if scores else float("nan"),
            "median": float(np.nanmedian(scores)) if scores else float("nan"),
        }
        for cls_id, scores in per_class_scores.items()
    }

    # global summaries
    all_scores = [d["score"] for d in video_scores.values()]
    avg_score   = float(np.nanmean(all_scores)) if all_scores else float("nan")
    max_score   = float(np.nanmax(all_scores))  if all_scores else float("nan")
    min_score   = float(np.nanmin(all_scores))  if all_scores else float("nan")
    median_score= float(np.nanmedian(all_scores)) if all_scores else float("nan")

    model.train()
    return video_scores, class_stats, avg_score, max_score, min_score, median_score


def make_stratified_train_loader(
    train_ds,
    clip_len: int,
    stride: int,
    pad_mode: str,
    stats,
    K_per_class: int = 64,         # target windows per class
    windows_per_video: int = 2,    # how many windows to take from each video
    seed: int = 1337,
    batch_size: int = 256,
):
    rng = random.Random(seed)
    grouped = defaultdict(list)  # cls_name -> list[(VideoItem, start)]

    # 1) pre-pass: count how many we still need per class
    need = {cls: K_per_class for cls in train_ds.classes}

    # 2) iterate videos; for each video, pick a few starts; add until class hits target
    for it in train_ds.items:
        cls = it.cls
        if need.get(cls, 0) <= 0:  # already satisfied
            continue

        # choose up to `windows_per_video` starts for this video
        if it.length <= clip_len:
            starts = [0]
        else:
            last = it.length - clip_len
            all_starts = list(range(0, last + 1, max(1, stride)))
            rng.shuffle(all_starts)
            starts = all_starts[:windows_per_video]

        for s in starts:
            if need[cls] <= 0:
                break
            grouped[cls].append((it, s))
            need[cls] -= 1

        # early exit if all classes satisfied
        if all(v <= 0 for v in need.values()):
            break

    # 3) flatten & build dataset/loader
    samples = [pair for pairs in grouped.values() for pair in pairs]
    ds_small = WindowDataset(
        samples, clip_len=clip_len, retries=0, jitter=0,
        pad_mode=pad_mode, seed=seed, stats=stats
    )
    loader_small = DataLoader(
        ds_small, batch_size=batch_size, shuffle=False,
        collate_fn=safe_collate, pin_memory=(torch.cuda.is_available())
    )
    return loader_small

@torch.no_grad()
def build_train_centroids_subset(model, small_loader, label_dict, device):
    model.eval()
    C = len(label_dict)
    sums, counts = None, torch.zeros(C, device=device, dtype=torch.float32)

    for packed in small_loader:
        if packed is None:
            continue
        feats, cls_names, _ = packed
        feats = feats.to(device, non_blocking=True)
        z, _, _ = model(feats)
        # z = F.normalize(z, dim=-1)

        y = torch.as_tensor([label_dict[c] for c in cls_names], device=device, dtype=torch.long)
        if sums is None:
            sums = torch.zeros(C, z.shape[1], device=device, dtype=torch.float32)

        sums.index_add_(0, y, z)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

    centroids = sums / counts.clamp_min(1.0).unsqueeze(1)
    centroids = F.normalize(centroids, dim=-1)
    model.train()
    return centroids, counts

@torch.no_grad()
def action_consistency_centroid(model, loader, label_dict, centroids, device):
    model.eval()
    centroids = F.normalize(centroids, dim=-1)
    id2cls = {i: c for c, i in label_dict.items()}
    C = centroids.size(0)

    from collections import defaultdict
    vid_accum = defaultdict(lambda: {"intra_sum": 0.0, "inter_sum": 0.0, "n": 0, "cls_id": None})

    for packed in loader:
        if packed is None: continue
        feats, cls_names, vids = packed
        feats = feats.to(device, non_blocking=True)
        z, _, _ = model(feats)
        z = F.normalize(z, dim=-1)

        y = torch.as_tensor([label_dict[c] for c in cls_names], device=device, dtype=torch.long)
        sims = z @ centroids.t()                # [B,C]
        d_all = 1.0 - sims                      # cosine distances

        b_idx = torch.arange(y.numel(), device=device)
        d_intra = d_all[b_idx, y]               # to own class centroid

        mask = torch.ones_like(d_all, dtype=torch.bool)
        mask[b_idx, y] = False
        d_inter = d_all.masked_fill(~mask, 0.0).sum(dim=1) / max(1, C - 1)

        for v, yi, di, do in zip(vids, y.tolist(), d_intra.tolist(), d_inter.tolist()):
            a = vid_accum[v]
            a["intra_sum"] += float(di)
            a["inter_sum"] += float(do)
            a["n"] += 1
            if a["cls_id"] is None:
                a["cls_id"] = yi

    # per-video
    video_scores = {}
    for v, a in vid_accum.items():
        n = max(1, a["n"])
        intra = a["intra_sum"] / n
        inter = a["inter_sum"] / n
        score = inter / (inter + intra) if (inter + intra) > 0 else float("nan")
        video_scores[v] = {"class": a["cls_id"], "windows": n, "intra": intra, "inter": inter, "score": score}

    # per-class
    per_class = defaultdict(list)
    for v, d in video_scores.items():
        per_class[d["class"]].append(d["score"])

    class_stats = {
        id2cls[k]: {
            "count_videos": len(vals),
            "avg": float(np.nanmean(vals)) if vals else float("nan"),
            "max": float(np.nanmax(vals))  if vals else float("nan"),
            "min": float(np.nanmin(vals))  if vals else float("nan"),
            "median": float(np.nanmedian(vals)) if vals else float("nan"),
        }
        for k, vals in per_class.items()
    }

    all_scores = [d["score"] for d in video_scores.values()]
    avg_score = float(np.nanmean(all_scores)) if all_scores else float("nan")
    max_score = float(np.nanmax(all_scores))  if all_scores else float("nan")
    min_score = float(np.nanmin(all_scores))  if all_scores else float("nan")
    median_score = float(np.nanmedian(all_scores)) if all_scores else float("nan")

    model.train()
    return video_scores, class_stats, avg_score, max_score, min_score, median_score


class PKBatchSamplerRequired(BatchSampler):
    """
    Metric-learning sampler: each batch has P classes, K samples/class.
    Additionally enforces that a given set of classes is included in *every* batch.
    - If a class runs out of fresh samples in the epoch, samples with replacement.
    - Shuffles per-class queues each epoch.

    Args:
        labels: sequence[int] length N (class id per sample)
        P: classes per batch
        K: samples per class
        required_class_ids: iterable[int], must be included in every batch
        drop_last: drop final incomplete batch (not used here because we control size)
        generator: numpy Generator for reproducibility (optional)
    """
    def __init__(
        self,
        labels: T.Sequence[int],
        P: int,
        K: int,
        required_class_ids: T.Optional[T.Sequence[int]] = None,
        drop_last: bool = False,
        generator=None,
    ):
        self.labels = np.asarray(labels)
        self.P = int(P)
        self.K = int(K)
        self.drop_last = drop_last
        self.rng = np.random.default_rng() if generator is None else generator

        # map: class_id -> indices
        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.class_to_indices[int(y)].append(idx)

        self.classes = list(self.class_to_indices.keys())

        # sanitize required classes
        self.required = sorted(set(required_class_ids or []))
        assert len(self.classes) >= self.P, "P must be <= number of classes present"
        assert self.P >= len(self.required), (
            f"P ({self.P}) must be >= number of required classes ({len(self.required)})"
        )
        missing = [c for c in self.required if c not in self.class_to_indices or len(self.class_to_indices[c]) == 0]
        if missing:
            raise ValueError(f"Required classes missing in labels: {missing}")

        self._reset_epoch()

    def _reset_epoch(self):
        # per-class shuffled queues
        self.per_class_queues = {}
        for c, idxs in self.class_to_indices.items():
            idxs = np.array(idxs)
            self.rng.shuffle(idxs)
            self.per_class_queues[c] = idxs.tolist()

        # classes available for the non-required slots
        self.non_required = [c for c in self.classes if c not in self.required]
        self.rng.shuffle(self.non_required)
        self.nr_cursor = 0

        # rough epoch length
        total_items = sum(len(v) for v in self.per_class_queues.values())
        self.num_batches = max(1, total_items // (self.P * self.K))

    def __len__(self):
        total_items = sum(len(v) for v in self.class_to_indices.values())
        return max(1, total_items // (self.P * self.K))

    def _take_from_class(self, c: int) -> T.List[int]:
        """Take K indices from class c (without replacement where possible, else top up with replacement)."""
        q = self.per_class_queues[c]
        if len(q) >= self.K:
            take = q[:self.K]
            del q[:self.K]
            return take
        else:
            take = q.copy()
            need = self.K - len(take)
            pool = self.class_to_indices[c]
            # replacement draw for the remainder
            if need > 0:
                take.extend(self.rng.choice(pool, size=need, replace=True).tolist())
            q.clear()
            return take

    def __iter__(self):
        self._reset_epoch()
        batches_emitted = 0

        while True:
            # Always include required classes
            chosen = list(self.required)

            # Fill remaining slots with non-required classes, cycling through shuffled list
            slots_left = self.P - len(chosen)
            if slots_left > 0:
                # wrap around if needed, reshuffle at every wrap
                picked = []
                while len(picked) < slots_left:
                    remaining = len(self.non_required) - self.nr_cursor
                    if remaining <= 0:
                        self.rng.shuffle(self.non_required)
                        self.nr_cursor = 0
                        remaining = len(self.non_required)
                    take = min(slots_left - len(picked), remaining)
                    picked.extend(self.non_required[self.nr_cursor:self.nr_cursor + take])
                    self.nr_cursor += take
                chosen.extend(picked)

            # sanity (unique classes)
            if len(set(chosen)) != self.P:
                # Just in case P almost equals total classes and required overlaps; repair by sampling uniques
                pool = [c for c in self.classes if c not in set(chosen)]
                self.rng.shuffle(pool)
                for c in pool:
                    if len(chosen) >= self.P: break
                    chosen.append(c)
                chosen = chosen[:self.P]

            # assemble batch
            batch = []
            for c in chosen:
                batch.extend(self._take_from_class(c))

            self.rng.shuffle(batch)
            if len(batch) != self.P * self.K:
                if self.drop_last:
                    continue
                # else, we still yield the (correct-sized) batch due to replacement top-up
            yield batch
            batches_emitted += 1

            if batches_emitted >= self.num_batches:
                break