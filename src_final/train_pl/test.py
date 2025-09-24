import os
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

# --------------------- local project imports ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import TemporalTransformerV2Plus
from losses import *
from utils import *
LATENT_DIM = 128
INPUT_DIM = 1250
PRIMARY_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class NpzVideoDatasetTest(Dataset):
    """
    Scans per-class directories for .npz files saved by save_video_npz(...).
    """
    def __init__(self, root_dir: str, items: T.Optional[T.List[VideoItem]] = None,
                 whitelist_json_dir: T.Optional[str] = None, filter_classes: T.Optional[T.List[str]] = None, min_videos_per_class: int = 10, enforce_min_per_class: bool = True):
        self.root_dir = root_dir
        self.filter_classes = filter_classes
        self.whitelist = self._load_whitelist(whitelist_json_dir) if whitelist_json_dir else {}

        raw_items = items if items is not None else self._scan(self.filter_classes)

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

    def _scan(self, filter_classes) -> T.List[VideoItem]:
        items: T.List[VideoItem] = []
        for cls in filter_classes:
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


FILTER_CLASSES = ["BodyWeightSquats", "JumpingJack", "HulaHoop", "PushUps", "PullUps"]

cogvideo_ds = NpzVideoDatasetTest("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz_cogvideox", filter_classes=FILTER_CLASSES)
print(f"cogvideo_ds: {len(cogvideo_ds)} samples")

gen4_ds = NpzVideoDatasetTest("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz_gen4", filter_classes=FILTER_CLASSES)
print(f"gen4_ds: {len(gen4_ds)} samples")

train_ds = NpzVideoDataset("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz")
# stats = compute_stats_from_npz(train_ds.items)

real_ds = NpzVideoDatasetTest("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz", filter_classes=FILTER_CLASSES)
print(f"real_ds: {len(real_ds)} samples")

# creat a subset of real_ds with only actions that are in cogvideo_ds and gen4_ds
common_actions = set(cogvideo_ds.classes) & set(gen4_ds.classes)
common_actions = common_actions & set(real_ds.classes)
print(f"common_actions: {len(common_actions)} actions")

from collections import OrderedDict

checkpoint = torch.load("/home/coder/projects/video_evals/video-gen-evals/src_final/train_pl/SAVE/temporal_transformer_best.pt", map_location=PRIMARY_DEVICE)

new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    # Remove 'module.' prefix if it exists
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

model = TemporalTransformerV2Plus(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(PRIMARY_DEVICE)
model.load_state_dict(new_state_dict, strict=False)


def sample_windows_npz(
    ds: NpzVideoDataset,
    clip_len: int = 32,
    stride: int = 1,
    windows_per_video: int = 4,
    seed: int = 1337
):
    """
    Returns a list of tuples: (VideoItem, start)
    """
    rng = random.Random(seed)
    vids = ds.items[:]
    rng.shuffle(vids)

    out: T.List[T.Tuple[VideoItem, int]] = []
    for it in vids:
        max_start = max(0, it.length - clip_len)
        if max_start <= 0:
            continue
        possible = list(range(0, max_start + 1, stride))
        # k = min(windows_per_video, len(possible))
        k = len(possible)
        starts = rng.sample(possible, k)
        for s in starts:
            out.append((it, s))
    rng.shuffle(out)
    return out
    
def balanced_windows_per_video(ds, clip_len=32, K=8, seed=1337):
    rng = random.Random(seed)
    out = []
    for it in ds.items:
        Tlen = it.length
        max_start = max(0, Tlen - clip_len)
        if max_start < 0:
            continue
        # if very short, allow dense starts; always have at least 1 candidate
        possible = list(range(0, max_start + 1, max(1, min(8, max_start or 1))))
        if len(possible) == 0:
            possible = [0]
        # sample exactly K starts WITH replacement (gives overlap for short clips)
        starts = [rng.choice(possible) for _ in range(K)]
        out.extend([(it, s) for s in starts])
    rng.shuffle(out)
    return out

windows = balanced_windows_per_video(
    ds=cogvideo_ds,
    clip_len=32,
    K=8,
    seed=1337
)
print(f"Total windows from cogvideo_ds: {len(windows)}")

cogvideox_windows = WindowDataset(windows, clip_len=32)
cogvideox_loader = DataLoader(cogvideox_windows, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
print(f"cogvideox_windows: {len(cogvideox_windows)} samples")

gen4_windows = balanced_windows_per_video(
    ds=gen4_ds,
    clip_len=32,
    K=8
)
print(f"Total windows from gen4_ds: {len(gen4_windows)}")

gen4x_windows = WindowDataset(gen4_windows, clip_len=32)
gen4x_loader = DataLoader(gen4x_windows, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
print(f"gen4x_windows: {len(gen4x_windows)} samples")


real_windows = balanced_windows_per_video(
    ds=real_ds,
    clip_len=32,
    K=8
)
real_windows = [w for w in real_windows if w[0].cls in common_actions]
real_windows_ds = WindowDataset(real_windows, clip_len=32)
real_loader = DataLoader(real_windows_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
print(f"real_windows_ds: {len(real_windows_ds)} samples")   

from collections import defaultdict
import numpy as np

# =========================
# 1) Collect window embeddings (FIX: accumulate lists, not overwrite)
# =========================
def collect_embeddings(dataloader, model, device=PRIMARY_DEVICE, desc=""):
    vid_to_embs = defaultdict(list)
    vid_to_cls  = {}
    model.eval()
    for batch in tqdm(dataloader, desc=desc):
        feats, cls_names, vids = batch
        feats = feats.to(device, non_blocking=True)
        with torch.no_grad():
            emb, _, _ = model(feats)  # [B, D]
        emb_np = emb.detach().cpu().numpy()
        for i in range(len(vids)):
            vid = vids[i]
            cls = cls_names[i]
            vid_to_embs[vid].append(emb_np[i])
            vid_to_cls[vid] = cls
    return vid_to_embs, vid_to_cls

cog_vid2embs, cog_vid2cls = collect_embeddings(cogvideox_loader, model, desc="CogVideoX windows")
gen4_vid2embs, gen4_vid2cls = collect_embeddings(gen4x_loader, model, desc="Gen4 windows")
real_vid2embs, real_vid2cls = collect_embeddings(real_loader, model, desc="Real windows")

# =========================
# 2) Aggregate windows -> one vector per video
# =========================
# def aggregate_video_embeddings(vid2embs, agg="mean"):
#     vid2vec = {}
#     for vid, embs in vid2embs.items():
#         E = np.stack(embs, axis=0)  # [num_windows, D]
#         if agg == "mean":
#             vid2vec[vid] = E.mean(axis=0)
#         elif agg == "median":
#             vid2vec[vid] = np.median(E, axis=0)
#         else:
#             raise ValueError("agg must be 'mean' or 'median'")
#     return vid2vec

# cog_vid2vec = aggregate_video_embeddings(cog_vid2embs, agg="median")
# gen4_vid2vec = aggregate_video_embeddings(gen4_vid2embs, agg="median")
# real_vid2vec = aggregate_video_embeddings(real_vid2embs, agg="median")


def compute_real_class_centroids(real_vid2embs, real_vid2cls, agg="mean"):
    """
    Compute reference centroids from real embeddings.
    Each class gets one centroid vector.
    """
    cls2all_embs = defaultdict(list)
    for vid, embs in real_vid2embs.items():
        cls = real_vid2cls[vid]
        E = np.stack(embs, axis=0)
        if agg == "mean":
            cls2all_embs[cls].append(E.mean(axis=0))
        elif agg == "median":
            cls2all_embs[cls].append(np.median(E, axis=0))
        else:
            raise ValueError("agg must be 'mean' or 'median'")

    cls2centroid = {}
    for cls, vecs in cls2all_embs.items():
        arr = np.stack(vecs, axis=0)
        cls2centroid[cls] = arr.mean(axis=0)  # final class centroid
    return cls2centroid

# build reference centroids from real data
ref_cls2centroid = compute_real_class_centroids(real_vid2embs, real_vid2cls, agg="mean")


def aggregate_video_embeddings_with_ref(vid2embs, ref_cls2centroid, vid2cls, mode="weighted"):
    """
    Aggregate per-video window embeddings, using real centroids as reference.

    Args:
        vid2embs: dict[vid] -> list of np.array[D]
        ref_cls2centroid: dict[class] -> np.array[D] (real class centroid in embedding space)
        vid2cls: dict[vid] -> str (class name for each video)
        mode: "weighted" or "mean"
            - "mean": plain average (sensitive to outliers already).
            - "weighted": weight windows by distance to real centroid
                          (farther = higher weight → more sensitive to outliers).
    """
    out = {}
    for vid, embs in vid2embs.items():
        E = np.stack(embs, axis=0)  # [num_windows, D]
        cls = vid2cls[vid]
        ref = ref_cls2centroid[cls]

        if mode == "mean":
            out[vid] = E.mean(axis=0)

        elif mode == "weighted":
            d = np.linalg.norm(E - ref[None, :], axis=1)  # [num_windows]
            # weights increase with distance from real centroid
            w = d / (d.sum() + 1e-8)
            out[vid] = (E * w[:, None]).sum(axis=0)

        else:
            raise ValueError("mode must be 'mean' or 'weighted'")
    return out

cog_vid2vec = aggregate_video_embeddings_with_ref(
    cog_vid2embs, ref_cls2centroid, cog_vid2cls, mode="weighted"
)

gen4_vid2vec = aggregate_video_embeddings_with_ref(
    gen4_vid2embs, ref_cls2centroid, gen4_vid2cls, mode="weighted"
)

real_vid2vec = aggregate_video_embeddings_with_ref(
    real_vid2embs, ref_cls2centroid, real_vid2cls, mode="mean"  # usually mean for reals
)

# Restrict to common actions
def filter_by_actions(vid2vec, vid2cls, allowed):
    out_v, out_c = {}, {}
    for vid, vec in vid2vec.items():
        cls = vid2cls[vid]
        if cls in allowed:
            out_v[vid] = vec
            out_c[vid] = cls
    return out_v, out_c

cog_vid2vec, cog_vid2cls = filter_by_actions(cog_vid2vec, cog_vid2cls, common_actions)
gen4_vid2vec, gen4_vid2cls = filter_by_actions(gen4_vid2vec, gen4_vid2cls, common_actions)
real_vid2vec, real_vid2cls = filter_by_actions(real_vid2vec, real_vid2cls, common_actions)

# =========================
# 3) Helpers: cosine/L2, MMD (median gamma), Frechet, grouping
# =========================
def l2_normalize(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / nrm

def cosine_similarity_matrix(A, B):
    A = l2_normalize(A)
    B = l2_normalize(B)
    return A @ B.T

def euclidean_dist_matrix(A, B):
    # A: [Na, D], B: [Nb, D] -> pairwise L2 distances
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True)
    AB = A @ B.T
    d2 = A2 + B2.T - 2*AB
    return np.sqrt(np.maximum(d2, 0.0))

def median_heuristic_gamma(X, Y=None):
    # gamma = 1 / median(||x - y||^2)
    if Y is None:
        Z = X
    else:
        Z = np.vstack([X, Y])
    if len(Z) > 2000:
        idx = np.random.choice(len(Z), size=2000, replace=False)
        Z = Z[idx]
    D = euclidean_dist_matrix(Z, Z)
    tri = D[np.triu_indices_from(D, k=1)]
    med = np.median(tri**2)
    return 1.0 / (med + 1e-8)

def rbf_kernel(X, Y, gamma):
    X2 = np.sum(X**2, axis=1, keepdims=True)
    Y2 = np.sum(Y**2, axis=1, keepdims=True)
    XY = X @ Y.T
    d2 = X2 + Y2.T - 2*XY
    return np.exp(-gamma * d2)

def compute_mmd_rbf(X, Y, gamma=None):
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return np.nan
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n*(n-1))
    term_yy = Kyy.sum() / (m*(m-1))
    term_xy = Kxy.mean()
    return term_xx + term_yy - 2*term_xy

def try_frechet(mu1, cov1, mu2, cov2):
    try:
        import scipy.linalg
        cov_prod_sqrt = scipy.linalg.sqrtm(cov1 @ cov2)
        if np.iscomplexobj(cov_prod_sqrt):
            cov_prod_sqrt = cov_prod_sqrt.real
        diff = mu1 - mu2
        return diff @ diff + np.trace(cov1 + cov2 - 2*cov_prod_sqrt)
    except Exception:
        return np.nan

def fit_gaussian(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(X) - 1 + 1e-8)
    return mu, cov

def by_action_arrays(vid2vec, vid2cls):
    act2arr = defaultdict(list)
    for vid, vec in vid2vec.items():
        act2arr[vid2cls[vid]].append(vec)
    for a in list(act2arr.keys()):
        act2arr[a] = np.stack(act2arr[a], axis=0)
    return act2arr

real_act = by_action_arrays(real_vid2vec, real_vid2cls)
cog_act  = by_action_arrays(cog_vid2vec, cog_vid2cls)
gen4_act = by_action_arrays(gen4_vid2vec, gen4_vid2cls)

# =========================
# 4) New L2 metrics (this is what you asked for)
# =========================
def mean_l2_distance(gen_arr, real_arr):
    D = euclidean_dist_matrix(gen_arr, real_arr)  # [Ng, Nr]
    return float(D.mean())

def real_intra_mean_l2(real_arr):
    if len(real_arr) < 2:
        return np.nan
    D = euclidean_dist_matrix(real_arr, real_arr)
    tri = D[np.triu_indices_from(D, k=1)]
    return float(tri.mean())

def normalized_mean_l2(gen_arr, real_arr):
    # ratio > 1 ⇒ gen is further from reals than reals are from each other
    num = mean_l2_distance(gen_arr, real_arr)
    den = real_intra_mean_l2(real_arr)
    return float(num / (den + 1e-8))

def centroid_cosine(gen_arr, real_arr):
    rc = real_arr.mean(axis=0, keepdims=True)
    S = cosine_similarity_matrix(gen_arr, rc)
    return float(S.mean())

# =========================
# 5) Per-action and overall metrics (incl. L2 + normalized L2)
# =========================
per_action = []
for act in sorted(common_actions):
    if act not in real_act:
        continue
    R = real_act[act]
    C = cog_act.get(act, None)
    G = gen4_act.get(act, None)

    muR, covR = fit_gaussian(R)
    row = {
        "action": act, "n_real": len(R), "cog_n": None, "gen4_n": None,
        # L2 metrics:
        "cog_mean_l2": np.nan, "gen4_mean_l2": np.nan,
        "cog_norm_l2": np.nan, "gen4_norm_l2": np.nan,
        # Cosine (kept for reference):
        "cog_centroid_cos": np.nan, "gen4_centroid_cos": np.nan,
        # Distribution distances:
        "cog_mmd": np.nan, "gen4_mmd": np.nan,
        "cog_frechet": np.nan, "gen4_frechet": np.nan
    }

    if C is not None and len(C) > 0:
        row["cog_n"] = len(C)
        row["cog_mean_l2"]   = mean_l2_distance(C, R)
        row["cog_norm_l2"]   = normalized_mean_l2(C, R)
        row["cog_centroid_cos"] = centroid_cosine(C, R)
        row["cog_mmd"] = compute_mmd_rbf(C, R, gamma=None)  # median heuristic
        muC, covC = fit_gaussian(C)
        row["cog_frechet"] = try_frechet(muR, covR, muC, covC)

    if G is not None and len(G) > 0:
        row["gen4_n"] = len(G)
        row["gen4_mean_l2"]   = mean_l2_distance(G, R)
        row["gen4_norm_l2"]   = normalized_mean_l2(G, R)
        row["gen4_centroid_cos"] = centroid_cosine(G, R)
        row["gen4_mmd"] = compute_mmd_rbf(G, R, gamma=None)
        muG, covG = fit_gaussian(G)
        row["gen4_frechet"] = try_frechet(muR, covR, muG, covG)

    per_action.append(row)

# Overall pooled
R_all = np.vstack([real_act[a] for a in real_act.keys()])
C_all = np.vstack([cog_act[a] for a in cog_act.keys()]) if len(cog_act) else None
G_all = np.vstack([gen4_act[a] for a in gen4_act.keys()]) if len(gen4_act) else None
muR_all, covR_all = fit_gaussian(R_all)

overall = {
    # L2 metrics:
    "cog_mean_l2": np.nan, "gen4_mean_l2": np.nan,
    "cog_norm_l2": np.nan, "gen4_norm_l2": np.nan,
    # Cosine (ref):
    "cog_centroid_cos": np.nan, "gen4_centroid_cos": np.nan,
    # Distribution:
    "cog_mmd": np.nan, "gen4_mmd": np.nan,
    "cog_frechet": np.nan, "gen4_frechet": np.nan
}
if C_all is not None and len(C_all) > 0:
    overall["cog_mean_l2"] = mean_l2_distance(C_all, R_all)
    overall["cog_norm_l2"] = normalized_mean_l2(C_all, R_all)
    overall["cog_centroid_cos"] = centroid_cosine(C_all, R_all)
    overall["cog_mmd"] = compute_mmd_rbf(C_all, R_all, gamma=None)
    muC_all, covC_all = fit_gaussian(C_all)
    overall["cog_frechet"] = try_frechet(muR_all, covR_all, muC_all, covC_all)

if G_all is not None and len(G_all) > 0:
    overall["gen4_mean_l2"] = mean_l2_distance(G_all, R_all)
    overall["gen4_norm_l2"] = normalized_mean_l2(G_all, R_all)
    overall["gen4_centroid_cos"] = centroid_cosine(G_all, R_all)
    overall["gen4_mmd"] = compute_mmd_rbf(G_all, R_all, gamma=None)
    muG_all, covG_all = fit_gaussian(G_all)
    overall["gen4_frechet"] = try_frechet(muR_all, covR_all, muG_all, covG_all)

# =========================
# 6) Global retrieval@1 with L2 (class match)
# =========================
def global_retrieval_at1_l2(gen_vid2vec, gen_vid2cls, real_vid2vec, real_vid2cls):
    if not len(gen_vid2vec) or not len(real_vid2vec):
        return np.nan
    gen_vids = list(gen_vid2vec.keys())
    real_vids = list(real_vid2vec.keys())
    G = np.stack([gen_vid2vec[v] for v in gen_vids], axis=0)
    R = np.stack([real_vid2vec[v] for v in real_vids], axis=0)
    D = euclidean_dist_matrix(G, R)  # [Ng, Nr]
    nn_idx = D.argmin(axis=1)        # nearest (smallest L2)
    correct = 0
    for i, rj in enumerate(nn_idx):
        if gen_vid2cls[gen_vids[i]] == real_vid2cls[real_vids[rj]]:
            correct += 1
    return correct / len(gen_vids)

cog_global_r1_l2  = global_retrieval_at1_l2(cog_vid2vec, cog_vid2cls, real_vid2vec, real_vid2cls)
gen4_global_r1_l2 = global_retrieval_at1_l2(gen4_vid2vec, gen4_vid2cls, real_vid2vec, real_vid2cls)

# =========================
# 7) Print tables (now includes L2 + normalized L2)
# =========================
print("\n=== OVERALL (all actions pooled) ===")
for k in ["cog_mean_l2","gen4_mean_l2","cog_norm_l2","gen4_norm_l2",
          "cog_centroid_cos","gen4_centroid_cos","cog_mmd","gen4_mmd","cog_frechet","gen4_frechet"]:
    v = overall[k]
    print(f"{k:>18}: {v:.6f}" if isinstance(v, float) and not np.isnan(v) else f"{k:>18}: {v}")

print("\n=== PER ACTION (includes L2) ===")
cols = ["action","n_real","cog_n","gen4_n",
        "cog_mean_l2","gen4_mean_l2","cog_norm_l2","gen4_norm_l2",
        "cog_centroid_cos","gen4_centroid_cos",
        "cog_mmd","gen4_mmd","cog_frechet","gen4_frechet"]
print("\t".join(cols))
for row in per_action:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.6f}" if not np.isnan(x) else "nan"
        return str(x)
    print("\t".join(fmt(row[c]) for c in cols))

print("\n=== GLOBAL RETRIEVAL@1 (L2, class match) ===")
print(f"CogVideoX: {cog_global_r1_l2:.6f}" if not np.isnan(cog_global_r1_l2) else "CogVideoX: nan")
print(f"Gen4     : {gen4_global_r1_l2:.6f}" if not np.isnan(gen4_global_r1_l2) else "Gen4     : nan")

# =========================
# PCA plotting utils (fixed & complete)
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except Exception as e:
    raise ImportError("scikit-learn is required for PCA plots. pip install scikit-learn") from e


# ---------- small helpers ----------
def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _stack_safe(arr_dict):
    """Stack dict[class] -> np.ndarray[N_c, D] into one [N, D] or return None."""
    return np.vstack([arr_dict[a] for a in arr_dict.keys()]) if len(arr_dict) else None

def _centroid(X):
    return X.mean(axis=0, keepdims=True)

def _euclid(a, b):
    d = a - b
    return float(np.sqrt((d**2).sum()))

def _plot_clouds_2d(real2, cog2=None, gen22=None, title="Embeddings (PCA-2D)", savepath=None,
                    s=12, alpha=0.35, show_centroids=True):
    plt.figure(figsize=(6, 6))

    # points
    if real2 is not None: plt.scatter(real2[:, 0], real2[:, 1], s=s, alpha=alpha, label="Real")
    if cog2  is not None: plt.scatter(cog2[:, 0],  cog2[:, 1],  s=s, alpha=alpha, label="CogVideoX")
    if gen22 is not None: plt.scatter(gen22[:, 0], gen22[:, 1], s=s, alpha=alpha, label="Gen4")

    # centroids + connectors
    if show_centroids and real2 is not None:
        cr = _centroid(real2)
        plt.scatter(cr[:, 0], cr[:, 1], marker="X", s=120, label="Real μ")
        if cog2 is not None:
            cc = _centroid(cog2)
            plt.scatter(cc[:, 0], cc[:, 1], marker="P", s=120, label="Cog μ")
            plt.plot([cr[0, 0], cc[0, 0]], [cr[0, 1], cc[0, 1]], linestyle="--", linewidth=1)
        if gen22 is not None:
            cg = _centroid(gen22)
            plt.scatter(cg[:, 0], cg[:, 1], marker="P", s=120, label="Gen4 μ")
            plt.plot([cr[0, 0], cg[0, 0]], [cr[0, 1], cg[0, 1]], linestyle="--", linewidth=1)

    # cosmetics
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, dpi=200)
    plt.show()


# ---------- whitening (optional but recommended) ----------
def _whiten_to_real_space(R_all, eps=1e-6):
    """
    Build whitening transform using ONLY real pooled video-level vectors.
    Returns (mu_R, W) such that X_whitened = (X - mu_R) @ W.T has ~I covariance
    if X ~ real distribution.

    R_all: np.ndarray [N_r, D]
    """
    mu = R_all.mean(axis=0)
    # note: np.cov expects features on rows when rowvar=True; we pass X^T
    cov = np.cov((R_all - mu).T)
    # regularize
    U, S, Vt = np.linalg.svd(cov + eps * np.eye(cov.shape[0]), full_matrices=False)
    W = (U @ np.diag(1.0 / np.sqrt(S)) @ U.T)  # cov^{-1/2}
    return mu, W


# ---------- main API ----------
def fit_pca_on_pooled(real_all, cog_all=None, gen4_all=None, whiten=False, random_state=0):
    """
    Fit a shared PCA basis on pooled data (optionally after whitening by real covariance).

    Inputs:
        real_all: [N_r, D] required
        cog_all : [N_c, D] or None
        gen4_all: [N_g, D] or None
        whiten  : bool, if True uses real-covariance whitening before PCA
        random_state: int for PCA reproducibility

    Returns:
        dict with keys:
            'pca'      : fitted PCA object (on raw or whitened space)
            'muR'      : real mean (only if whiten=True, else None)
            'W'        : whitening matrix (only if whiten=True, else None)
            'R2','C2','G2': 2D projections for real/cog/gen4 (may be None)
    """
    assert real_all is not None and len(real_all), "real_all must be non-empty"

    if whiten:
        muR, W = _whiten_to_real_space(real_all)
        Rw = (real_all - muR) @ W.T
        Cw = None if cog_all is None else (cog_all - muR) @ W.T
        Gw = None if gen4_all is None else (gen4_all - muR) @ W.T
        pooled = Rw if (Cw is None and Gw is None) else np.vstack([x for x in [Rw, Cw, Gw] if x is not None])
        pca = PCA(n_components=2, random_state=random_state).fit(pooled)
        R2 = pca.transform(Rw)
        C2 = None if Cw is None else pca.transform(Cw)
        G2 = None if Gw is None else pca.transform(Gw)
        return {"pca": pca, "muR": muR, "W": W, "R2": R2, "C2": C2, "G2": G2}
    else:
        pooled = real_all if (cog_all is None and gen4_all is None) else np.vstack([x for x in [real_all, cog_all, gen4_all] if x is not None])
        pca = PCA(n_components=2, random_state=random_state).fit(pooled)
        R2 = pca.transform(real_all)
        C2 = None if cog_all is None else pca.transform(cog_all)
        G2 = None if gen4_all is None else pca.transform(gen4_all)
        return {"pca": pca, "muR": None, "W": None, "R2": R2, "C2": C2, "G2": G2}


def plot_overall_and_per_action(
    real_act, cog_act, gen4_act, out_dir="plots",
    whiten=False, random_state=0, s=12, alpha=0.35, show_centroids=True,
    print_header=True
):
    """
    real_act / cog_act / gen4_act: dict[action] -> np.ndarray[n_action_videos, D]
        (These should be *video-level* embeddings, not window-level.)
    """
    # overall (pooled)
    R_all = _stack_safe(real_act)
    C_all = _stack_safe(cog_act)
    G_all = _stack_safe(gen4_act)

    fitted = fit_pca_on_pooled(R_all, C_all, G_all, whiten=whiten, random_state=random_state)
    R2, C2, G2 = fitted["R2"], fitted["C2"], fitted["G2"]

    # overall centroid gaps (PCA space)
    cr = _centroid(R2)
    if print_header:
        print(f"=== PCA (whiten={whiten}) overall centroid gaps ===")
    if C2 is not None:
        cc = _centroid(C2)
        print(f"[OVERALL] ||μ_real - μ_cog||_PCA2D = {_euclid(cr, cc):.3f}")
    if G2 is not None:
        cg = _centroid(G2)
        print(f"[OVERALL] ||μ_real - μ_gen4||_PCA2D = {_euclid(cr, cg):.3f}")

    # overall plot
    _plot_clouds_2d(R2, C2, G2, title=f"Pooled actions: PCA-2D (whiten={whiten})",
                    savepath=os.path.join(out_dir, f"overall_pca2d_whiten_{int(whiten)}.png"),
                    s=s, alpha=alpha, show_centroids=show_centroids)

    # per-action plots (use the SAME PCA basis)
    # If whitening was used, transform each action set with the same (muR, W), then PCA.transform
    pca = fitted["pca"]
    muR, W = fitted["muR"], fitted["W"]

    if print_header:
        print("\n=== PCA per-action centroid gaps ===")
    actions = sorted(set(real_act.keys()) | set(cog_act.keys()) | set(gen4_act.keys()))
    for act in actions:
        R = real_act.get(act, None)
        C = cog_act.get(act, None)
        G = gen4_act.get(act, None)
        if R is None:  # need real for the reference cloud
            continue

        if whiten:
            Rw = (R - muR) @ W.T
            Cw = None if C is None else (C - muR) @ W.T
            Gw = None if G is None else (G - muR) @ W.T
            R2 = pca.transform(Rw)
            C2 = None if Cw is None else pca.transform(Cw)
            G2 = None if Gw is None else pca.transform(Gw)
        else:
            R2 = pca.transform(R)
            C2 = None if C is None else pca.transform(C)
            G2 = None if G is None else pca.transform(G)

        # centroid gap printout
        cr = _centroid(R2)
        msg = [f"[{act}]"]
        if C2 is not None: msg.append(f"Δμ(real,cog)={_euclid(cr, _centroid(C2)):.3f}")
        if G2 is not None: msg.append(f"Δμ(real,gen4)={_euclid(cr, _centroid(G2)):.3f}")
        print("  ".join(msg))

        # plot
        _plot_clouds_2d(R2, C2, G2,
                        title=f"{act}: PCA-2D (whiten={whiten})",
                        savepath=os.path.join(out_dir, f"{act}_pca2d_whiten_{int(whiten)}.png"),
                        s=s, alpha=alpha, show_centroids=show_centroids)

# Example calls (do both for paper-ready visuals):
plot_overall_and_per_action(real_act, cog_act, gen4_act, out_dir="plots", whiten=False, random_state=0)
plot_overall_and_per_action(real_act, cog_act, gen4_act, out_dir="plots", whiten=True,  random_state=0)