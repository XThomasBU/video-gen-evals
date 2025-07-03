import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate

# ——— Configuration ———
CODEBOOK_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/codebook.npy"
JSD_EPS       = 1e-8

# ——— Load codebook ———
print("Loading codebook...")
codebook = np.load(CODEBOOK_PATH)
print(f"Codebook shape: {codebook.shape}")

def quantize(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)      # (N, 1)
    codebook_norm_sq = np.sum(codebook ** 2, axis=1)       # (K,)
    distances = x_norm_sq - 2 * np.dot(x, codebook.T) + codebook_norm_sq
    return np.argmin(distances, axis=1)

# ——— Token I/O ———
def load_single_token(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    toks = np.array(data.get("discrete_token", []))
    if toks.ndim > 1:
        toks = toks[0].flatten()
    return toks

def load_single_cls_logits_softmax(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    arr = np.array(data.get("cls_logits_softmax", []))
    if arr.ndim > 1:
        arr = arr[0]
    return arr

def load_video_tokens(video_folder):
    npz = Path(video_folder) / "tokens.npz"
    if not npz.exists():
        frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
        all_toks = [load_single_token(p) for p in frames]
        np.savez_compressed(npz, tokens=all_toks)
    return list(np.load(npz, allow_pickle=True)["tokens"])

def load_cls_logits_softmax(video_folder):
    npz = Path(video_folder) / "cls_logits_softmax.npz"
    if not npz.exists():
        frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
        all_out = [load_single_cls_logits_softmax(p) for p in frames]
        np.savez_compressed(npz, cls_logits_softmax=all_out)
    try:
        return list(np.load(npz, allow_pickle=True)["cls_logits_softmax"])
    except:
        frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
        all_out = [load_single_cls_logits_softmax(p) for p in frames]
        np.savez_compressed(npz, cls_logits_softmax=all_out)
        return list(np.load(npz, allow_pickle=True)["cls_logits_softmax"])

def apply_temperature(logits, T=0.5):
    logits = logits / T
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

# ——— Build a softmax-based transition histogram ———
def compute_softmax_transition_histogram(cls_logits_softmax_seq, top_k=5):
    """
    Computes a (flattened) transition histogram using softmax logits.
    Uses only top-K entries in each softmax to reduce noise and sparsity.
    """
    transition_matrix = np.zeros((2048, 2048), dtype=np.float32)

    for s1, s2 in zip(cls_logits_softmax_seq[:-1], cls_logits_softmax_seq[1:]):
        if s1.size == 0 or s2.size == 0:
            continue
        # s1, s2 shape = (160, 2048)
        s1 = apply_temperature(s1, T=0.5)
        s2 = apply_temperature(s2, T=0.5)
        s1_topk = topk_filter(s1, top_k)
        s2_topk = topk_filter(s2, top_k)

        transition_matrix += s1_topk.T @ s2_topk

    # flat = transition_matrix.flatten()
    # return flat / flat.sum() if flat.sum() > 0 else flat
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums
    return transition_matrix.flatten()

def topk_filter(matrix, k):
    """
    For each row in the (N, 2048) matrix,
    zero out all but top-k entries, and renormalize to sum to 1.
    """
    out = np.zeros_like(matrix)
    for i, row in enumerate(matrix):
        if k >= len(row):
            out[i] = row
            continue
        topk_idx = np.argpartition(-row, k)[:k]
        topk_vals = row[topk_idx]
        topk_vals = np.maximum(topk_vals, 0)
        if topk_vals.sum() > 0:
            out[i, topk_idx] = topk_vals / topk_vals.sum()
    return out

def compute_transition_jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))

# ——— Compute per-class lists of transition histograms ———
def compute_class_transition_embeddings(all_videos):
    class_transitions = defaultdict(list)
    for vid in tqdm(all_videos, desc="Embedding transitions"):
        hist = compute_softmax_transition_histogram(vid["cls_logits_softmax"])
        class_transitions[vid["class"]].append(hist)
    return class_transitions

# ——— Compute leave-one-out intra/inter statistics ———
def compute_inter_intra_jsd(class_transitions):
    intra_jsds, inter_jsds = [], []
    class_names = list(class_transitions.keys())
    for cls in class_names:
        own = class_transitions[cls]
        others = [h for c in class_names if c!=cls for h in class_transitions[c]]
        for hist in own:
            # intra vs its class-mates
            intra = [compute_transition_jsd(hist, h2) for h2 in own if not np.allclose(hist,h2)]
            if intra: intra_jsds.append(np.mean(intra))
            # inter vs all other classes
            inter = [compute_transition_jsd(hist, h2) for h2 in others]
            if inter: inter_jsds.append(np.mean(inter))
    return {
        'intra_jsd_mean': np.mean(intra_jsds), 'intra_jsd_std': np.std(intra_jsds),
        'inter_jsd_mean': np.mean(inter_jsds), 'inter_jsd_std': np.std(inter_jsds),
        'intra_jsds': intra_jsds, 'inter_jsds': inter_jsds
    }

# ——— New: pairwise class vs class average JSD matrix ———
def compute_class_vs_class_jsd_table(class_transitions):
    class_names = sorted(class_transitions.keys())
    table = []
    for A in class_names:
        row = []
        for B in class_names:
            jsd_vals = [ compute_transition_jsd(a, b)
                         for a in class_transitions[A]
                         for b in class_transitions[B] ]
            row.append(np.mean(jsd_vals) if jsd_vals else 0.0)
        table.append(row)
    return class_names, table

def print_jsd_table(class_names, table):
    headers = [""] + class_names
    rows = []
    for name,row in zip(class_names, table):
        rows.append([name] + [f"{v:.4f}" for v in row])
    print("\n=== Class-vs-Class Average JSD ===")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

# ——— Main ———
if __name__ == "__main__":
    REAL_ROOT     = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
    GEN_ROOT      = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos"
    all_videos = []
    classes = os.listdir(REAL_ROOT)
    # classes = classes[3:5]
    for cls_dir in sorted(classes):
        folder = Path(REAL_ROOT)/cls_dir
        if not folder.is_dir(): continue
        for vid in sorted(os.listdir(folder)):
            seq = load_video_tokens(folder/vid)
            logits = load_cls_logits_softmax(folder/vid)
            if seq and logits:
                all_videos.append({"class": cls_dir, "video": vid, "cls_logits_softmax": logits})

    print(f"Loaded {len(all_videos)} videos total.")
    class_transitions = compute_class_transition_embeddings(all_videos)

    jsd_stats = compute_inter_intra_jsd(class_transitions)
    print(f"\nIntra-class JSD: {jsd_stats['intra_jsd_mean']:.4f} ± {jsd_stats['intra_jsd_std']:.4f}")
    print(f"Inter-class JSD: {jsd_stats['inter_jsd_mean']:.4f} ± {jsd_stats['inter_jsd_std']:.4f}")

    # print the new pairwise table
    names, matrix = compute_class_vs_class_jsd_table(class_transitions)
    print_jsd_table(names, matrix)