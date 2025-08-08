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

def partial_shuffle_within_window(seqs, lengths, vid_ids, shuffle_fraction=0.5):
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