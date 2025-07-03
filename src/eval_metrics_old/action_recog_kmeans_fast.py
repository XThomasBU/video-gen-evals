import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import entropy
import pickle
from tabulate import tabulate

# ——— Configuration ———
CODEBOOK_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/codebook.npy"
JSD_EPS       = 1e-8
TOP_K         = 5
TEMPERATURE   = 0.5

# ——— Load codebook ———
print("Loading codebook...")
codebook = np.load(CODEBOOK_PATH)
print(f"Codebook shape: {codebook.shape}")

# ——— Token I/O ———
def load_single_cls_logits_softmax(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    arr = np.array(data.get("cls_logits_softmax", []))
    if arr.ndim > 1:
        arr = arr[0]
    return arr

def load_video_tokens(video_folder):
    npz_path = Path(video_folder) / "tokens_logits.npz"
    data = np.load(npz_path, allow_pickle=True)
    return list(data["tokens"])

def load_cls_logits_softmax(video_folder):
    npy = Path(video_folder) / "cls_logits_softmax.npy"
    if npy.exists():
        return list(np.load(npy, allow_pickle=True))
    # else:
    #     frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    #     all_out = [load_single_cls_logits_softmax(p) for p in tqdm(frames, desc="Loading frames")]
    #     np.save(npy, np.array(all_out, dtype=object))
    #     return all_out

# ——— Softmax + top-K + temperature ———
def apply_temperature(token_logits, T=TEMPERATURE):
    for token in range(token_logits.shape[0]):
        logits = token_logits[token] / T
        logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        logits = logits / logits.sum(axis=-1, keepdims=True)
        token_logits[token] = logits
    return token_logits

def topk_filter(matrix, k=TOP_K):
    out = np.zeros_like(matrix)
    for i, row in enumerate(matrix):
        if k >= len(row):
            out[i] = row
        else:
            idx = np.argpartition(-row, k)[:k]
            vals = np.maximum(row[idx], 0)
            if vals.sum() > 0:
                out[i, idx] = vals / vals.sum()
    return out

# ——— Transition histogram ———
def compute_softmax_transition_histogram(seq):
    seq = np.array(seq, dtype=np.float32)
    M = 2048 #seq[0].shape[1]
    hist = np.zeros((M, M), dtype=np.float32)
    for s1, s2 in zip(seq[:-1], seq[1:]):
        if s1.size == 0 or s2.size == 0:
            continue
        s1_t = topk_filter(apply_temperature(s1), TOP_K)
        s2_t = topk_filter(apply_temperature(s2), TOP_K)
        # s1_t = topk_filter(s1, TOP_K)
        # s2_t = topk_filter(s2, TOP_K)
        hist += s1_t.T @ s2_t
    # normalize rows
    rowsum = hist.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    hist /= rowsum
    return hist.flatten()

def compute_descrete_transition_histogram(token_seq, vocab_size=2048):
    hist = np.zeros((vocab_size, vocab_size))
    for i in range(len(token_seq) - 1):
        a, b = token_seq[i], token_seq[i + 1]
        for ta, tb in zip(a, b):
            hist[ta, tb] += 1
    flat = hist.flatten()
    return flat / flat.sum() if flat.sum() > 0 else flat

# ——— Batched JSD computation ———
def compute_batch_jsd_matrix(P, Q, eps=1e-8):
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    M = 0.5 * (P[:, None, :] + Q[None, :, :])
    M = np.clip(M, eps, 1)

    KL_P_M = np.sum(P[:, None, :] * np.log2(P[:, None, :] / M), axis=-1)
    KL_Q_M = np.sum(Q[None, :, :] * np.log2(Q[None, :, :] / M), axis=-1)
    return 0.5 * (KL_P_M + KL_Q_M)

# ——— Collect histograms ———
def collect_class_histograms(videos, type='softmax'):
    ct = defaultdict(list)
    for vid in tqdm(videos, desc="Computing histograms"):
        if type == 'softmax':
            hist = compute_softmax_transition_histogram(vid['cls_logits_softmax'])
        elif type == 'descrete':
            hist = compute_descrete_transition_histogram(vid['tokens'])
        ct[vid['class']].append({'video': vid['video'], 'hist': hist})
    return ct

# ——— Intra/inter statistics with batching ———
def compute_inter_intra_jsd(class_hists):
    intra, inter = [], []
    for cls, items in class_hists.items():
        own = np.stack([i['hist'] for i in items])
        others = np.stack([h['hist'] for c, hs in class_hists.items() if c != cls for h in hs])

        if len(own) > 1:
            jsd_matrix_intra = compute_batch_jsd_matrix(own, own)
            # exclude self-diagonal
            mask = ~np.eye(jsd_matrix_intra.shape[0], dtype=bool)
            intra_vals = jsd_matrix_intra[mask]
            intra.append(np.mean(intra_vals))

        if len(others) > 0:
            jsd_matrix_inter = compute_batch_jsd_matrix(own, others)
            inter.append(np.mean(jsd_matrix_inter))

    return {
        'intra_mean': np.mean(intra),
        'intra_std': np.std(intra),
        'inter_mean': np.mean(inter),
        'inter_std': np.std(inter)
    }

# ——— Pairwise table ———
def compute_pairwise_table(class_hists):
    names = sorted(class_hists.keys())
    table = []
    all_hists = {k: np.stack([i['hist'] for i in v]) for k, v in class_hists.items()}

    for A in tqdm(names, desc="Computing pairwise table"):
        row = []
        for B in names:
            mat = compute_batch_jsd_matrix(all_hists[A], all_hists[B])
            row.append(np.mean(mat) if mat.size else 0.0)
        table.append(row)
    return names, table

# ——— Rank best/median/worst ———
def rank_videos_per_class(class_hists, reference='intra'):
    rankings = {}
    for cls, items in class_hists.items():
        own = np.stack([i['hist'] for i in items])
        others = np.stack([
            h['hist']
            for c, hs in class_hists.items()
            if (c != cls if reference == 'intra' else c == cls)
            for h in hs
        ])
        if len(others) == 0:
            continue

        jsd_matrix = compute_batch_jsd_matrix(own, others)
        median_scores = np.median(jsd_matrix, axis=1)
        sorted_idx = np.argsort(median_scores)

        best = items[sorted_idx[0]]['video']
        worst = items[sorted_idx[-1]]['video']
        med = items[sorted_idx[len(sorted_idx) // 2]]['video']
        rankings[cls] = {'best': best, 'median': med, 'worst': worst}
    return rankings

# ——— Print JSD table ———
def print_jsd_table(class_names, table):
    headers = [""] + class_names
    rows = []
    for name, row in zip(class_names, table):
        rows.append([name] + [f"{v:.4f}" for v in row])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def compute_idf(real):
    token_frame_counts = {i: 0 for i in range(2048)}
    total_frames = 0

    for video in tqdm(real, desc="Computing IDF"):
        for frame_tokens in video['tokens']:
            total_frames += 1
            # Treat this frame as a document: only unique tokens matter
            unique_tokens = set(frame_tokens)
            for t in unique_tokens:
                token_frame_counts[t] += 1

    idf = {t: np.log((1 + total_frames) / (1 + token_frame_counts[t])) for t in token_frame_counts}
    return idf


def tfidf_coverage(videoA, videoB, idf):
    # All unique tokens
    tokens_A = set()
    for frame_tokens in videoA['tokens']:
        tokens_A.update(frame_tokens)

    tokens_B = set()
    for frame_tokens in videoB['tokens']:
        tokens_B.update(frame_tokens)

    # Overlap
    overlap = tokens_A.intersection(tokens_B)
    union = tokens_A.union(tokens_B)

    overlap_weight = sum(idf.get(t, 0.0) for t in overlap)
    union_weight = sum(idf.get(t, 0.0) for t in union)

    if union_weight == 0:
        return 0.0

    coverage = overlap_weight / union_weight
    return coverage


# ——— Main flow ———
def main(real_root, gen_root):
    def load_all_(root, tag):
        vids = []
        classes = os.listdir(root)
        for cls in classes:
            p = Path(root) / cls
            if not p.is_dir(): continue
            for vid in tqdm(sorted(os.listdir(p)), desc="Loading videos"):
                logits = load_cls_logits_softmax(p / vid)
                tokens = load_video_tokens(p / vid)
                if logits:
                    vids.append({'class': cls, 'video': vid, 'cls_logits_softmax': logits, 'type': tag, 'tokens': tokens})
        return vids
    def load_all(root, tag):
        vids = []
        classes = os.listdir(root)
        classes = ['WallPushups', 'PushUps']
        for cls in classes:
            p = Path(root) / cls
            if not p.is_dir(): continue
            for vid in tqdm(sorted(os.listdir(p)), desc="Loading videos"):
                logits = load_cls_logits_softmax(p / vid)
                tokens = load_video_tokens(p / vid)
                if logits:
                    vids.append({'class': cls, 'video': vid, 'cls_logits_softmax': logits, 'type': tag, 'tokens': tokens})
        return vids

    real = load_all(real_root, 'real')
    gen = load_all(gen_root, 'gen')
    real_all = load_all_(real_root, 'real')
    print(f"Loaded {len(real)} real videos, {len(gen)} generated videos")

    real_idf = compute_idf(real_all)
    coverage_test = tfidf_coverage(real_all[0], real_all[1], real_idf)
    print(coverage_test)
    exit()

    real_h = collect_class_histograms(real, type='softmax')
    gen_h = collect_class_histograms(gen, type='softmax')
    

    for label, ch in [('REAL', real_h), ('GEN', gen_h)]:
        stats = compute_inter_intra_jsd(ch)
        print(f"=== {label} VIDEOS STATISTICS ===")
        print(f"Intra-class JSD: {stats['intra_mean']:.4f} ± {stats['intra_std']:.4f}")
        print(f"Inter-class JSD: {stats['inter_mean']:.4f} ± {stats['inter_std']:.4f}\n")

    for label, ch in [('REAL', real_h), ('GENERATED', gen_h)]:
        print(f"=== {label} VIDEOS: Class-vs-Class Average JSD ===")
        names, tbl = compute_pairwise_table(ch)
        print_jsd_table(names, tbl)

    print("=== REAL vs GENERATED COMPARISON ===")
    rows = []
    for cls in sorted(real_h):
        rlist = np.stack([v['hist'] for v in real_h[cls]])
        glist = np.stack([v['hist'] for v in gen_h.get(cls, [])])
        if len(glist) == 0:
            continue
        jsd_matrix = compute_batch_jsd_matrix(rlist, glist)
        rows.append([
            cls,
            rlist.shape[0],
            glist.shape[0],
            f"{np.mean(jsd_matrix):.4f}",
            f"{np.std(jsd_matrix):.4f}"
        ])
    print(tabulate(rows, headers=["Class", "Real#", "Gen#", "MeanJSD", "StdJSD"], tablefmt="grid"))

    print("\n=== BEST/MEDIAN/WORST VIDEOS (by intra-class JSD) ===")
    rnk = rank_videos_per_class(gen_h, 'intra')
    for cls, info in rnk.items():
        print(f"{cls}: best={info['best']}, median={info['median']}, worst={info['worst']}")

if __name__ == "__main__":
    REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
    GEN_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_wan21_videos"
    main(REAL_ROOT, GEN_ROOT)