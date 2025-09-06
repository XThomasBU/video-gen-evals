# ============================
# NEW: Unified latent plot for
# real + ALL generative models
# ============================

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from test2 import *
from save_data import  extract_windows

from train import (
    # TemporalTransformer,
    # load_video_sequence,
    # extract_windows,
    # collate_fn,
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)
from models import TemporalTransformerV2Plus
from utils import *

# ==============================
# SO(3) helpers (rotation utils)
# ==============================

def _axis_angle_to_matrix(a: torch.Tensor) -> torch.Tensor:
    """Axis-angle -> rotation matrix via Rodrigues.
    a: [..., 3]
    returns: [..., 3, 3]
    """
    theta = a.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # [..., 1]
    k = a / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    O = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([O,   -kz,  ky], dim=-1),
        torch.stack([kz,   O,  -kx], dim=-1),
        torch.stack([-ky,  kx,   O], dim=-1),
    ], dim=-2)  # [..., 3, 3]

    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(a.shape[:-1] + (3, 3))

    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]

    return I + s * K + (1.0 - c) * (K @ K)


def _log_so3(R: torch.Tensor) -> torch.Tensor:
    """Matrix log on SO(3) -> axis-angle vector.
    R: [..., 3, 3]
    returns: [..., 3]
    """
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1 + 1e-6, 3 - 1e-6)
    theta = torch.acos((tr - 1) / 2)
    denom = (2 * torch.sin(theta)).unsqueeze(-1).clamp_min(1e-6)
    v = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) / denom
    return theta.unsqueeze(-1) * v


# ===================
# Modality deltas
# ===================

def _vit_delta(vit: torch.Tensor) -> torch.Tensor:
    """Cosine-stable feature change.
    vit: [T, D]
    returns: [T, D]
    """
    v = F.normalize(vit, dim=-1)
    v_prev = torch.cat([v[:1], v[:-1]], dim=0)
    return v - v_prev


def _rot_axisangle_delta(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle pose -> SO(3) relative delta via log map.
    aa: [T, 3*J]
    returns: [T, 3*J] (axis-angle deltas per joint)
    """
    T, D = aa.shape
    J = D // 3
    a = aa.view(T, J, 3)
    a_prev = torch.cat([a[:1], a[:-1]], dim=0)
    R = _axis_angle_to_matrix(a)
    R0 = _axis_angle_to_matrix(a_prev)
    Rrel = torch.matmul(R0.transpose(-1, -2), R)
    w = _log_so3(Rrel)
    return w.view(T, D)


def _rot_matrix_delta(Rflat: torch.Tensor) -> torch.Tensor:
    """Relative rotation between consecutive frames in 9D matrix form.
    Rflat: [T, 9]
    returns: [T, 9]
    """
    T = Rflat.shape[0]
    R = Rflat.view(T, 3, 3)
    R0 = torch.cat([R[:1], R[:-1]], dim=0)
    Rrel = torch.matmul(R0.transpose(-1, -2), R)
    return Rrel.reshape(T, 9)


def _procrustes_kp_delta(kp: torch.Tensor) -> torch.Tensor:
    """Root/scale-normalized keypoint velocity.
    kp: [T, 2*K]
    returns: [T, 2*K]
    """
    T, D = kp.shape
    pts = kp.view(T, -1, 2)
    pts_c = pts - pts.mean(dim=1, keepdim=True)
    s = torch.linalg.norm(pts_c, dim=(1, 2), keepdim=True).clamp_min(1e-6)
    pts_n = pts_c / s
    prev = torch.cat([pts_n[:1], pts_n[:-1]], dim=0)
    return (pts_n - prev).reshape(T, D)


def _betas_delta(betas: torch.Tensor, ema: float = 0.9, max_abs: float = 0.1) -> torch.Tensor:
    """EMA-smoothed shape change (mostly near-zero).
    betas: [T, B]
    returns: [T, B]
    """
    diff = betas - torch.cat([betas[:1], betas[:-1]], dim=0)
    out = torch.zeros_like(diff)
    acc = torch.zeros((1, betas.shape[1]), device=betas.device, dtype=betas.dtype)
    for t in range(betas.shape[0]):
        acc = ema * acc + (1 - ema) * diff[t:t+1]
        out[t:t+1] = acc
    return out.clamp(-max_abs, max_abs)


def load_video_sequence(video_folder, MESH_DIR, POSE_DIR, stats_dir):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace(MESH_DIR, POSE_DIR)
    twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

    stats_dir = Path(stats_dir)
    keys_raw = ['vit', 'global_orient', 'body_pose', 'betas', 'twod_kp']
    keys_d   = ['vit_d', 'global_orient_d', 'body_pose_d', 'betas_d', 'twod_kp_d']

    # load stats (RAW + MOTION)
    stats = {}
    for k in keys_raw:
        stats[f'{k}_mean'] = np.load(stats_dir / f'{k}_mean.npy')
        stats[f'{k}_std']  = np.load(stats_dir / f'{k}_std.npy')
    for k in keys_d:
        stats[f'{k}_mean'] = np.load(stats_dir / f'{k}_mean.npy')
        stats[f'{k}_std']  = np.load(stats_dir / f'{k}_std.npy')

    T = min(len(frames), len(twod_points_paths))
    if T < 2:
        return None

    vit_list, go_list, pose_list, betas_list, kp_list = [], [], [], [], []
    for t in range(T):
        with open(frames[t], "rb") as f:
            data = pickle.load(f)
        params = data["pred_smpl_params"]
        if isinstance(params, list):
            if len(params) < 1:
                continue
            params = params[0]
        if not isinstance(params, dict):
            continue

        vit  = np.asarray(params["token_out"]).reshape(-1)[:1024]
        go   = np.asarray(params["global_orient"]).reshape(-1)[:9]
        pose = np.asarray(params["body_pose"]).reshape(-1)[:207]
        bet  = np.asarray(params["betas"]).reshape(-1)[:10]
        kp   = np.load(twod_points_paths[t]).reshape(-1)[:120]

        vit_list.append(torch.tensor(vit, dtype=torch.float32))
        go_list.append(torch.tensor(go, dtype=torch.float32))
        pose_list.append(torch.tensor(pose, dtype=torch.float32))
        betas_list.append(torch.tensor(bet, dtype=torch.float32))
        kp_list.append(torch.tensor(kp, dtype=torch.float32))

    # [T,dim] tensors (RAW, unnormalized)
    vit   = torch.stack(vit_list, dim=0)
    go    = torch.stack(go_list, dim=0)
    pose  = torch.stack(pose_list, dim=0)
    betas = torch.stack(betas_list, dim=0)
    kp2d  = torch.stack(kp_list, dim=0)

    # ---------- RAW z-score ----------
    def z(x, mean, std):
        return (x - torch.tensor(mean, dtype=x.dtype)) / (torch.tensor(std, dtype=x.dtype) + 1e-8)

    # print(vit.shape, stats['vit_mean'].shape, stats['vit_std'].shape)
    vit_raw   = z(vit,   stats['vit_mean'],           stats['vit_std'])
    go_raw    = z(go,    stats['global_orient_mean'], stats['global_orient_std'])
    pose_raw  = z(pose,  stats['body_pose_mean'],     stats['body_pose_std'])
    betas_raw = z(betas, stats['betas_mean'],         stats['betas_std'])
    kp_raw    = z(kp2d,  stats['twod_kp_mean'],       stats['twod_kp_std'])

    raw = torch.cat([vit_raw, go_raw, pose_raw, betas_raw, kp_raw], dim=-1)

    # ---------- MOTION: delta first (on raw geometry), then z-score ----------
    d_vit  = _vit_delta(vit)
    d_go   = _rot_matrix_delta(go)
    d_pose = _rot_axisangle_delta(pose)
    d_bet  = _betas_delta(betas)
    d_kp   = _procrustes_kp_delta(kp2d)

    d_vit_n  = z(d_vit,  stats['vit_d_mean'],            stats['vit_d_std'])
    d_go_n   = z(d_go,   stats['global_orient_d_mean'],  stats['global_orient_d_std'])
    d_pose_n = z(d_pose, stats['body_pose_d_mean'],      stats['body_pose_d_std'])
    d_bet_n  = z(d_bet,  stats['betas_d_mean'],          stats['betas_d_std'])
    d_kp_n   = z(d_kp,   stats['twod_kp_d_mean'],        stats['twod_kp_d_std'])

    motion = torch.cat([d_vit_n, d_go_n, d_pose_n, d_bet_n, d_kp_n], dim=-1)

    assert raw.shape == motion.shape, f"raw {raw.shape} != motion {motion.shape}"
    enriched = torch.cat([raw, motion], dim=-1)

    # # Sanity check on expected dims (optional)
    # if expect_dim is not None:
    #     assert raw.shape[-1] == expect_dim, f"Expected raw dim {expect_dim}, got {raw.shape[-1]}"

    return enriched  # [T, 2*expect_dim]

INPUT_DIM= 1370

# -------- Config --------
ALL_GEN_MODELS = ["wan21", "runway_gen4", "hunyuan_360p", "opensora_256p", "cogvideox"]
METHOD = "tsne"  # "pca" or "tsne"
FIG_OUT = f"SAVE_TEST/all_models_latents_2d_{METHOD}.png"
FIG_SIZE = (11, 9)
BATCH_SIZE = 64

# Colors per source (feel free to tweak)
COLOR_MAP = {
    "real": "#111111",
    "wan21": "#1f77b4",
    "runway_gen4": "#ff7f0e",
    "hunyuan_360p": "#2ca02c",
    "opensora_256p": "#d62728",
    "cogvideox": "#9467bd",
    "centroid": "#000000",
}

# ----- Helper: build per-video mean embeddings for REAL dataset -----
def collect_real_video_means(all_train_embeds: torch.Tensor,
                             all_train_labels: torch.Tensor,
                             all_train_vid_ids: np.ndarray):
    """
    Returns:
      vid_names: list[str]
      vid_embeds: np.ndarray [N_videos, D]
      vid_classes: np.ndarray [N_videos]
      sources: list[str] = "real" for each
    """
    vid_means = {}
    vid_cls = {}
    # compute per-video mean
    for vid in np.unique(all_train_vid_ids):
        mask = (all_train_vid_ids == vid)
        emb_mean = all_train_embeds[mask].mean(dim=0)  # [D]
        # pick the majority class in that video
        cls = torch.mode(all_train_labels[mask])[0].item()
        vid_means[str(vid)] = emb_mean.cpu().numpy()
        vid_cls[str(vid)] = cls

    vid_names = list(vid_means.keys())
    vid_embeds = np.stack([vid_means[v] for v in vid_names], axis=0)
    vid_classes = np.array([vid_cls[v] for v in vid_names])
    sources = ["real"] * len(vid_names)
    return vid_names, vid_embeds, vid_classes, sources

# ----- Helper: load/make generated dataset same as your pipeline -----
class GeneratedVideoDataset(Dataset):
    def __init__(self, root, classes, window_size, stride):
        self.samples, self.labels, self.video_names = [], [], []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_dir = Path(root) / cls
            if not cls_dir.exists():
                continue
            for vid in os.listdir(cls_dir):
                seq = load_video_sequence(Path(root) / cls / vid, MESH_DIR, POSE_DIR, "SAVE")
                if seq is None:
                    continue
                wins = extract_windows(seq, window_size, stride)
                self.samples.extend(wins)
                self.labels.extend([self.class_to_idx[cls]] * len(wins))
                self.video_names.extend([vid] * len(wins))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.video_names[idx]

# ----- Helper: per-model per-video mean embeddings -----
def collect_generated_video_means(MODEL: str,
                                  all_classes,
                                  input_dim_twice,
                                  latent_dim,
                                  device,
                                  window_size=32,
                                  stride=8):
    """
    Returns:
      vid_names: list[str]
      vid_embeds: np.ndarray [N_videos, D]
      vid_classes: np.ndarray [N_videos]
      sources: list[str] = MODEL for each
    Uses your same TemporalTransformerV2Plus and load_video_sequence path logic.
    """
    # Resolve dirs like your loop
    GEN_ROOT = f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_{MODEL}_videos"

    # pick per-model POSE_DIR/MESH_DIR like your if/elif chain
    if "cogvideox" in GEN_ROOT:
        POSE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_COGVIDEOX"
        MESH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos"
    elif "opensora" in GEN_ROOT:
        POSE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_OPENSORA_256p"
        MESH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_opensora_256p_videos"
    elif "hunyuan" in GEN_ROOT:
        POSE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_HUNYUAN_360p"
        MESH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_hunyuan_360p_videos"
    elif "runway" in GEN_ROOT:
        if MODEL == "runway_gen3_alpha":
            POSE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN3_ALPHA"
            MESH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen3_alpha_videos"
        else:
            POSE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN4"
            MESH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos"
    elif "wan21" in GEN_ROOT:
        POSE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_WAN21"
        MESH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_wan21_videos"
    else:
        POSE, MESH = None, None

    # Expose globals used by load_video_sequence
    global POSE_DIR, MESH_DIR
    POSE_DIR, MESH_DIR = POSE, MESH

    # Load model once per call (shares weights with your training)
    trans = TemporalTransformerV2Plus(input_dim=input_dim_twice, latent_dim=latent_dim).to(device)
    state_path = f"SAVE/temporal_transformer_model_window_32_stride_8_valid_window_NO_ENT.pt"
    trans.load_state_dict(torch.load(state_path, map_location=device))
    trans.eval()

    # ds = GeneratedVideoDataset(GEN_ROOT, all_classes, window_size, stride)
    ds = torch.load(f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train_FINAL2backup/SAVE_TEST/gen_dataset_window_32_stride_8_valid_window_MODEL_{MODEL}.pt", weights_only=False)
    if len(ds) == 0:
        return [], np.zeros((0, latent_dim)), np.zeros((0,), dtype=int), []

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=lambda x: (
            torch.stack([i[0] for i in x]),
            torch.tensor([i[1] for i in x]),
            [i[2] for i in x],
        ),
    )

    # Aggregate per-video mean
    per_vid_embs = defaultdict(list)
    per_vid_cls = {}
    with torch.no_grad():
        for seqs, labels, vid_names in loader:
            seqs = seqs.to(device)
            lengths = torch.full((seqs.shape[0],), seqs.shape[1], dtype=torch.long, device=device)
            embs, _, _ = trans(seqs, lengths)   # [B, D]
            embs = embs.cpu()
            for e, c, v in zip(embs, labels, vid_names):
                per_vid_embs[v].append(e.numpy())
                per_vid_cls[v] = int(c)

    vids = list(per_vid_embs.keys())
    vid_means = np.stack([np.mean(per_vid_embs[v], axis=0) for v in vids], axis=0)
    vid_classes = np.array([per_vid_cls[v] for v in vids])
    sources = [MODEL] * len(vids)
    return vids, vid_means, vid_classes, sources


WINDOW_SIZE = 32
STRIDE = 8
 # ————— load train embeddings & centroids —————
all_train_embeds = torch.load(f"SAVE/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
all_train_labels = torch.load(f"SAVE/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
all_train_vid_ids = torch.load(f"SAVE/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt", weights_only=False)

# ---------- Gather everything ----------
print("\n[ALL-MODELS] Collecting REAL per-video means...")
real_vids, real_embs, real_classes, real_sources = collect_real_video_means(
    all_train_embeds, all_train_labels, all_train_vid_ids.numpy() if hasattr(all_train_vid_ids, "numpy") else all_train_vid_ids
)

all_vids = real_vids[:]
all_embs = [real_embs]
all_classes_list = [real_classes]
all_sources = real_sources[:]

for MODEL in ALL_GEN_MODELS:
    print(f"[ALL-MODELS] Collecting GEN per-video means for {MODEL}...")
    vids, embs, classes, sources = collect_generated_video_means(
        MODEL, ALL_CLASSES, INPUT_DIM*2, LATENT_DIM, DEVICE, window_size=32, stride=8
    )
    all_vids.extend(vids)
    all_embs.append(embs)
    all_classes_list.append(classes)
    all_sources.extend(sources)

all_embs = np.vstack(all_embs)                                # [N_total, D]
all_classes_arr = np.concatenate(all_classes_list, axis=0)    # [N_total]
all_sources_arr = np.array(all_sources)                       # [N_total]

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

# Optionally include class centroids (from your earlier computation)
centroid_keys_sorted = sorted(centroids.keys())
centroid_mat = torch.stack([centroids[k].cpu() for k in centroid_keys_sorted]).numpy()  # [C, D]

# L2-normalize before 2D projection (helps TSNE/PCA stability)
all_embs_norm = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-9)
centroids_norm = centroid_mat / (np.linalg.norm(centroid_mat, axis=1, keepdims=True) + 1e-9)

# Combine for joint projection
combined = np.concatenate([all_embs_norm, centroids_norm], axis=0)

print(f"[ALL-MODELS] Projecting with {METHOD.upper()}...")
if METHOD.lower() == "tsne":
    projector = TSNE(n_components=2, perplexity=35, learning_rate="auto", init="pca", random_state=42)
else:
    projector = PCA(n_components=2, random_state=42)

proj = projector.fit_transform(combined)
pts = proj[: len(all_embs_norm)]
cent_pts = proj[len(all_embs_norm):]

# ---------- Plot ----------
plt.figure(figsize=FIG_SIZE)

# Plot by source (real + each model) in different colors
sources_unique = ["real"] + [m for m in ALL_GEN_MODELS if m in all_sources_arr]
for src in sources_unique:
    mask = (all_sources_arr == src)
    if not np.any(mask): 
        continue
    plt.scatter(
        pts[mask, 0], pts[mask, 1],
        s=18, alpha=0.75, label=f"{src}",
        color=COLOR_MAP.get(src, None), edgecolors="none"
    )

# Plot centroids (black X)
plt.scatter(cent_pts[:, 0], cent_pts[:, 1],
            s=160, marker="X", linewidths=1.5, edgecolors="k",
            facecolors=COLOR_MAP["centroid"], label="class centroids")

plt.title(f"Real vs Generated Video Embeddings — {METHOD.upper()} Projection")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True, alpha=0.25)
plt.legend(loc="best", fontsize=9, ncol=1)
plt.tight_layout()
os.makedirs("SAVE_TEST", exist_ok=True)
plt.savefig(FIG_OUT, dpi=220)
plt.close()
print(f"[ALL-MODELS] Saved: {FIG_OUT}")

import os, json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.linalg import sqrtm

os.makedirs("SAVE_TEST/metrics", exist_ok=True)

def _to_np(x):
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    return np.asarray(x)

# --- Split by source (real vs each gen) & class ---
def split_by_source_and_class(embs, sources, classes):
    buckets = defaultdict(list)  # (source, class_idx) -> list of np vectors
    for e, s, c in zip(embs, sources, classes):
        buckets[(s, int(c))].append(e)
    # convert to arrays
    out = {k: np.stack(v, axis=0) if len(v) else np.zeros((0, embs.shape[1])) for k, v in buckets.items()}
    return out

# --- Basic distances ---
def pairwise_dists(A, B):
    # A: [na,d], B:[nb,d] -> [na,nb] euclidean
    if len(A)==0 or len(B)==0:
        return np.zeros((len(A), len(B)))
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(A2 + B2 - 2*A@B.T, 0.0))

def mahalanobis_dists(X, mu, cov, eps=1e-6):
    # X:[n,d], mu:[d], cov:[d,d]
    d = cov.shape[0]
    cov_reg = cov + eps*np.eye(d)
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)
    diff = X - mu[None, :]
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

def frechet_gaussian(m1, C1, m2, C2, eps=1e-6):
    # FID-style distance^2 between Gaussians N(m1,C1) and N(m2,C2)
    diff = m1 - m2
    diff2 = diff @ diff
    # product sqrt (may be slightly non-PSD -> stabilize)
    covmean, _ = np.linalg.eigh(C1 @ C2)
    # If eigh used above, alternative is sqrtm; here we use sqrtm for clarity:
    covmean = sqrtm(C1 @ C2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr = np.trace(C1 + C2 - 2*covmean)
    return float(diff2 + tr)

# --- Set metrics ---
def coverage(real, gen, epsilon):
    # fraction of real points within epsilon of some gen point
    if len(real)==0 or len(gen)==0: return 0.0
    D = pairwise_dists(real, gen)
    return float(np.mean(np.min(D, axis=1) <= epsilon))

def minimum_matching_distance(real, gen):
    # directed MMD: mean over real of min distance to gen (and reverse)
    if len(real)==0 or len(gen)==0: return np.nan
    D = pairwise_dists(real, gen)
    fwd = float(np.mean(np.min(D, axis=1)))
    bwd = float(np.mean(np.min(D, axis=0)))
    return (fwd + bwd)/2.0, fwd, bwd

def knn_realness(gen, real, k=5):
    if len(real) == 0 or len(gen) == 0: return np.nan
    D = pairwise_dists(gen, real)
    k = min(k, D.shape[1])
    nn = np.partition(D, kth=k-1, axis=1)[:, :k]
    return float(nn.mean())


all_embs_np = np.asarray(all_embs)
buckets = split_by_source_and_class(all_embs_np, all_sources_arr, all_classes_arr)

# 2) Precompute real stats per class
real_stats = {}
for c in sorted(set(all_classes_arr.tolist())):
    real_key = ("real", c)
    R = buckets.get(real_key, np.zeros((0, all_embs_np.shape[1])))
    mu = R.mean(axis=0) if len(R) else np.zeros((all_embs_np.shape[1],))
    if len(R) > 1:
        C = np.cov(R.T)
    else:
        C = np.eye(all_embs_np.shape[1])
    real_stats[c] = {"mu": mu, "cov": C, "count": len(R)}

# 3) Compute per-model/per-class metrics
models = sorted(set(all_sources_arr.tolist()) - {"real"})
rows = []
for model in models:
    for c in sorted(real_stats.keys()):
        real_key = ("real", c)
        gen_key  = (model, c)
        R = buckets.get(real_key, np.zeros((0, all_embs_np.shape[1])))
        G = buckets.get(gen_key, np.zeros((0, all_embs_np.shape[1])))

        mu_r, C_r, n_r = real_stats[c]["mu"], real_stats[c]["cov"], real_stats[c]["count"]
        if len(G) == 0:
            rows.append({
                "model": model, "class": c, "n_real": n_r, "n_gen": 0,
                "centroid_L2": np.nan, "mahalanobis_mean": np.nan,
                "kNN5_realness": np.nan, "coverage_eps@p95real": np.nan,
                "MMD_set": np.nan, "MMD_fwd": np.nan, "MMD_bwd": np.nan,
                "frechet": np.nan, "p95_real_to_gen": np.nan, "hausdorff_dir": np.nan
            })
            continue

        # centroid distance
        mu_g = G.mean(axis=0)
        centroid_L2 = float(np.linalg.norm(mu_g - mu_r))

        # Mahalanobis (mean)
        maha = mahalanobis_dists(G, mu_r, C_r)
        maha_mean = float(np.mean(maha)) if len(maha) else np.nan

        # kNN realness (k=5) in euclidean space
        knn5 = knn_realness(G, R, k=5)

        # coverage at epsilon = 95th percentile of real->real NN distance
        if len(R) > 1:
            Dr = pairwise_dists(R, R)
            np.fill_diagonal(Dr, np.inf)
            rnn = np.min(Dr, axis=1)
            eps = float(np.percentile(rnn, 95))
            cov_eps = coverage(R, G, eps)
        else:
            eps, cov_eps = np.nan, np.nan

        # set MMD (mean min dist) both directions
        mmd_avg, mmd_fwd, mmd_bwd = minimum_matching_distance(R, G)

        # Fréchet (Gaussian) using real stats and gen stats
        if len(G) > 1:
            C_g = np.cov(G.T)
            fre = frechet_gaussian(mu_r, C_r, mu_g, C_g)
        else:
            fre = np.nan

        # percentile / directed Hausdorff (real -> gen)
        if len(R) and len(G):
            D_rg = pairwise_dists(R, G)
            p95_rg = float(np.percentile(np.min(D_rg, axis=1), 95))
            haus_rg = float(np.max(np.min(D_rg, axis=1)))  # directed
        else:
            p95_rg, haus_rg = np.nan, np.nan

        rows.append({
            "model": model, "class": c, "n_real": n_r, "n_gen": len(G),
            "centroid_L2": centroid_L2, "mahalanobis_mean": maha_mean,
            "kNN5_realness": knn5, "coverage_eps@p95real": cov_eps,
            "MMD_set": mmd_avg, "MMD_fwd": mmd_fwd, "MMD_bwd": mmd_bwd,
            "frechet": fre, "p95_real_to_gen": p95_rg, "hausdorff_dir": haus_rg
        })

df = pd.DataFrame(rows)
df_sorted = df.sort_values(["model","class"]).reset_index(drop=True)
print("\n📊 Distance-based metrics (per model × class):")
print(df_sorted.head(20))

agg = df_sorted.groupby("model").agg({
    "centroid_L2":"mean",
    "mahalanobis_mean":"mean",
    "kNN5_realness":"mean",
    "coverage_eps@p95real":"mean",
    "MMD_set":"mean",
    "frechet":"mean",
    "p95_real_to_gen":"mean",
    "hausdorff_dir":"mean"
}).reset_index()
print("\n📈 Aggregated (mean over classes) per model:")
print(agg)