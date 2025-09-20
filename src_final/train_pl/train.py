# # -*- coding: utf-8 -*-
# """
# NPZ-window training (no frames, no HMR). Loads pre-saved arrays:
#   pose[T,69], betas[T,10], global_orient[T,3], vit[T,D], frame_idx[T], meta:str

# - Samples fixed-length windows from per-class .npz files
# - Concatenates appearance + motion deltas per frame
# - Trains TemporalTransformerV2 + ArcMargin head

# Author: you 🫡
# """

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
from models import TemporalTransformerV2
from losses import *
from utils import *  # partial_shuffle_within_window, reverse_sequence, get_static_window, etc.

# --------------------------- Determinism ---------------------------

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator(); g.manual_seed(SEED)

# ------------------------------ Config -----------------------------

DATASET_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/meshes_npz"
# Expect layout: DATASET_DIR/<class>/<video>.npz produced by save_video_npz(...)

WHITELIST_JSON_DIR = None  # set to per-class whitelist jsons if needed

BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 90
CLIP_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NGPUS = torch.cuda.device_count()
USE_DP = (DEVICE == "cuda") and (NGPUS > 1)
PRIMARY_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")

TOTAL_WINDOWS_PER_EPOCH = 256
WINDOWS_PER_VIDEO = 4
STRIDE = 1

RETRIES = 2          # retries are cheaper here (arrays); 2 is usually enough
JITTER = CLIP_LEN // 2
PAD_MODE = "repeat"  # repeat nearest frame if window lands near edges

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

def _betas_delta(betas: torch.Tensor, ema: float = 0.9, max_abs: float = 0.1) -> torch.Tensor:
    """EMA-smoothed shape change. betas: [T,B] -> [T,B]"""
    diff = betas - torch.cat([betas[:1], betas[:-1]], dim=0)
    out = torch.zeros_like(diff)
    acc = torch.zeros((1, betas.shape[1]), device=betas.device, dtype=betas.dtype)
    for t in range(betas.shape[0]):
        acc = ema * acc + (1 - ema) * diff[t:t+1]
        out[t:t+1] = acc
    return out.clamp(-max_abs, max_abs)

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
                 whitelist_json_dir: T.Optional[str] = WHITELIST_JSON_DIR):
        self.root_dir = root_dir
        self.whitelist = self._load_whitelist(whitelist_json_dir) if whitelist_json_dir else {}
        if items is not None:
            self.items = items
        else:
            self.items = self._scan()

        # class lists
        self.classes = sorted(list({it.cls for it in self.items}))
        self.class_to_items: T.Dict[str, T.List[VideoItem]] = {}
        for it in self.items:
            self.class_to_items.setdefault(it.cls, []).append(it)

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
        split_idx = int(len(vids_copy) * train_ratio)
        train_items.extend(vids_copy[:split_idx])
        test_items.extend(vids_copy[split_idx:])
    return NpzVideoDataset(dataset.root_dir, items=train_items, whitelist_json_dir=None), \
           NpzVideoDataset(dataset.root_dir, items=test_items,  whitelist_json_dir=None)

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
                 seed: int = 1337):
        self.samples = samples
        self.clip_len = clip_len
        self.retries = retries
        self.jitter = jitter
        self.pad_mode = pad_mode
        self.rng = random.Random(seed)

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
        # vit: cosine-stable diff in embedding space
        vit_diff = _vit_delta(vit_t)                # [T, D]

        # betas: linear finite diff (no EMA)
        beta_prev = torch.cat([beta_raw[:1], beta_raw[:-1]], dim=0)
        beta_diff = (beta_raw - beta_prev).clamp(-0.2, 0.2)   # [T,10]

        # rotations: group-aware relative rotation, flattened as (R_rel - I) -> 9 per rotation
        I3 = torch.eye(3, dtype=pose_R.dtype, device=pose_R.device)

        # global orient
        gori_prev = torch.cat([gori_R[:1], gori_R[:-1]], dim=0)              # [T,3,3]
        gori_rel  = torch.matmul(gori_prev.transpose(-1, -2), gori_R)        # [T,3,3]
        gori_diff = (gori_rel - I3).reshape(T, -1)                           # [T,9]

        # per-joint pose
        pose_prev = torch.cat([pose_R[:1], pose_R[:-1]], dim=0)              # [T,J,3,3]
        pose_rel  = torch.matmul(pose_prev.transpose(-1, -2), pose_R)        # [T,J,3,3]
        pose_diff = (pose_rel - I3).reshape(T, -1)                           # [T,9*J]=207

        # ---------- pack in model's modality order ----------
        # Order must match TemporalTransformerV2Plus.modalities: ["vit", "global", "pose", "beta"]
        raw  = torch.cat([vit_raw,  gori_raw,  pose_raw,  beta_raw], dim=-1)
        diff = torch.cat([vit_diff, gori_diff, pose_diff, beta_diff], dim=-1)

        feats = torch.cat([raw, diff], dim=-1)  # [T, total_in]  (RAW | MOTION)
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

# # ============================== Training ==============================

# # Build dataset + splits
# full_ds = NpzVideoDataset(DATASET_DIR, whitelist_json_dir=WHITELIST_JSON_DIR)
# train_ds, test_ds = train_test_split(full_ds, train_ratio=0.8, seed=SEED)
# print(f"Train videos: {len(train_ds)}, Test videos: {len(test_ds)}")

# ALL_CLASSES = sorted(list({it.cls for it in full_ds.items}))
# label_dict = {cls: i for i, cls in enumerate(ALL_CLASSES)}

# # Peek one window to determine feature dim
# probe_samples = sample_windows_capped_npz(train_ds, clip_len=CLIP_LEN, stride=STRIDE,
#                                           windows_per_video=1, total_cap=1, seed=SEED)
# assert len(probe_samples) > 0, "No training windows found."
# probe_ds = WindowDataset(probe_samples, clip_len=CLIP_LEN, retries=0, jitter=0, pad_mode=PAD_MODE)
# probe = probe_ds[0]
# assert probe is not None
# probe_feats, _, _ = probe
# INPUT_DIM = probe_feats.shape[-1]
# print(f"Derived input_dim per frame = {INPUT_DIM}")

# # Create model/head on primary device
# model = TemporalTransformerV2(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(PRIMARY_DEVICE)
# arc = ArcMarginProduct(LATENT_DIM, len(ALL_CLASSES), s=30.0, m=0.35).to(PRIMARY_DEVICE)

# if USE_DP:
#     print(f"Using DataParallel across {NGPUS} GPUs: {list(range(NGPUS))}")
#     model = torch.nn.DataParallel(model, device_ids=list(range(NGPUS)))
#     arc = torch.nn.DataParallel(arc, device_ids=list(range(NGPUS)))

# loss_fn = TCL().to(PRIMARY_DEVICE)
# loss_hard = SupConWithHardNegatives().to(PRIMARY_DEVICE)

# params = list(model.parameters()) + list(arc.parameters())
# optimizer = torch.optim.AdamW(params, lr=3e-4)
# # Per-step cosine: approximate steps/epoch from TOTAL_WINDOWS_PER_EPOCH / BATCH_SIZE
# steps_per_epoch = max(1, math.ceil(TOTAL_WINDOWS_PER_EPOCH / max(1, BATCH_SIZE)))
# cosine_anneal = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=steps_per_epoch * EPOCHS, eta_min=1e-6
# )



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
from utils import *  # partial_shuffle_within_window, reverse_sequence, get_static_window, etc.

# --------------------------- Determinism ---------------------------

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator(); g.manual_seed(SEED)

# ------------------------------ Config -----------------------------

DATASET_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/meshes_npz"
# Expect layout: DATASET_DIR/<class>/<video>.npz produced by save_video_npz(...)

WHITELIST_JSON_DIR = None  # set to per-class whitelist jsons if needed

BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 90
CLIP_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NGPUS = torch.cuda.device_count()
USE_DP = (DEVICE == "cuda") and (NGPUS > 1)
PRIMARY_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")

TOTAL_WINDOWS_PER_EPOCH = 2048
WINDOWS_PER_VIDEO = 8
STRIDE = 8

RETRIES = 2          # retries are cheaper here (arrays); 2 is usually enough
JITTER = CLIP_LEN // 2
PAD_MODE = "repeat"  # repeat nearest frame if window lands near edges


# ============================== Training ==============================

# Build dataset + splits
full_ds = NpzVideoDataset(DATASET_DIR, whitelist_json_dir=WHITELIST_JSON_DIR)
train_ds, test_ds = train_test_split(full_ds, train_ratio=0.8, seed=SEED)
print(f"Train videos: {len(train_ds)}, Test videos: {len(test_ds)}")

ALL_CLASSES = sorted(list({it.cls for it in full_ds.items}))
label_dict = {cls: i for i, cls in enumerate(ALL_CLASSES)}

# Peek one window to determine feature dim
probe_samples = sample_windows_capped_npz(train_ds, clip_len=CLIP_LEN, stride=STRIDE,
                                          windows_per_video=1, total_cap=1, seed=SEED)
assert len(probe_samples) > 0, "No training windows found."
probe_ds = WindowDataset(probe_samples, clip_len=CLIP_LEN, retries=0, jitter=0, pad_mode=PAD_MODE)
probe = probe_ds[0]
assert probe is not None
probe_feats, _, _ = probe
INPUT_DIM = probe_feats.shape[-1]
print(f"Derived input_dim per frame = {INPUT_DIM}")

# Create model/head on primary device
model = TemporalTransformerV2Plus(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(PRIMARY_DEVICE)
arc = ArcMarginProduct(LATENT_DIM, len(ALL_CLASSES), s=30.0, m=0.35).to(PRIMARY_DEVICE)

if USE_DP:
    print(f"Using DataParallel across {NGPUS} GPUs: {list(range(NGPUS))}")
    model = torch.nn.DataParallel(model, device_ids=list(range(NGPUS)))
    arc = torch.nn.DataParallel(arc, device_ids=list(range(NGPUS)))

loss_fn = TCL().to(PRIMARY_DEVICE)
loss_hard = SupConWithHardNegatives().to(PRIMARY_DEVICE)

params = list(model.parameters()) + list(arc.parameters())
optimizer = torch.optim.AdamW(params, lr=3e-4)
# Per-step cosine: approximate steps/epoch from TOTAL_WINDOWS_PER_EPOCH / BATCH_SIZE
steps_per_epoch = max(1, math.ceil(TOTAL_WINDOWS_PER_EPOCH / max(1, BATCH_SIZE)))
cosine_anneal = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=steps_per_epoch * EPOCHS, eta_min=1e-6
)

def make_test_loader(test_ds: NpzVideoDataset,
                     clip_len: int,
                     stride: int,
                     windows_per_video: int = 2,
                     total_cap: int = 1024,
                     seed: int = 999,
                     batch_size: int = 256):
    samples = sample_windows_capped_npz(
        test_ds,
        clip_len=clip_len,
        stride=stride,
        windows_per_video=windows_per_video,
        total_cap=total_cap,
        seed=seed,
    )
    ds = WindowDataset(samples, clip_len=clip_len, retries=0, jitter=0, pad_mode=PAD_MODE, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        worker_init_fn=seed_worker, generator=g, collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda"),
    )
@torch.no_grad()
def action_consistency(model, loader, label_dict, device=PRIMARY_DEVICE):
    model.eval()
    all_z, all_y = [], []

    for packed in loader:
        if packed is None: 
            continue
        feats, cls_names, _ = packed
        feats = feats.to(device, non_blocking=True)
        z, _, _ = model(feats)                   # [B, LATENT_DIM]
        z = F.normalize(z, dim=-1)               # normalize for cosine distance
        y = torch.as_tensor([label_dict[c] for c in cls_names],
                            device=z.device, dtype=torch.long)
        all_z.append(z)
        all_y.append(y)

    if not all_z:
        return float("nan"), float("nan"), float("nan"), {}

    Z = torch.cat(all_z, dim=0)                  # [N, D]
    Y = torch.cat(all_y, dim=0)                  # [N]

    # cosine distances
    S = Z @ Z.t()                                # similarity [N, N]
    D = 1.0 - S                                  # distance
    N = D.shape[0]
    eye = torch.eye(N, device=D.device, dtype=torch.bool)

    # global masks
    same = (Y.unsqueeze(0) == Y.unsqueeze(1)) & ~eye
    diff = ~same

    # global means
    intra_vals = D[same]
    inter_vals = D[diff]
    intra_mean = intra_vals.mean().item() if intra_vals.numel() > 0 else float("nan")
    inter_mean = inter_vals.mean().item() if inter_vals.numel() > 0 else float("nan")
    score = inter_mean / (inter_mean + intra_mean) if (
        np.isfinite(intra_mean) and np.isfinite(inter_mean) and
        (inter_mean + intra_mean) > 0
    ) else float("nan")

    # ---------- per-class stats ----------
    inv_label_dict = {v: k for k, v in label_dict.items()}
    class_stats = {}

    class_scores = []
    for cls_id in torch.unique(Y):
        mask_c = (Y == cls_id)
        idx_c = mask_c.nonzero(as_tuple=False).squeeze(-1)

        if idx_c.numel() < 2:
            # skip classes with <2 samples
            continue

        Dc = D[idx_c][:, idx_c]   # intra-class submatrix
        De = D[idx_c][:, ~mask_c] # cross-class distances

        # drop diagonal from intra
        intra_vals_c = Dc[~torch.eye(len(idx_c), device=D.device, dtype=torch.bool)]
        inter_vals_c = De.reshape(-1)

        intra_c = intra_vals_c.mean().item() if intra_vals_c.numel() > 0 else float("nan")
        inter_c = inter_vals_c.mean().item() if inter_vals_c.numel() > 0 else float("nan")
        score_c = inter_c / (inter_c + intra_c) if (
            np.isfinite(intra_c) and np.isfinite(inter_c) and
            (inter_c + intra_c) > 0
        ) else float("nan")

        class_stats[inv_label_dict[int(cls_id.item())]] = {
            "intra": intra_c,
            "inter": inter_c,
            "score": score_c,
            "count": int(idx_c.numel())
        }
        class_scores.append(score_c)

    avg_score = np.mean(class_scores) if class_scores else float("nan")
    max_score = np.max(class_scores) if class_scores else float("nan")
    min_score = np.min(class_scores) if class_scores else float("nan")
    median_score = np.median(class_scores) if class_scores else float("nan")
    num_classes_great_90 = sum(1 for s in class_scores if np.isfinite(s) and s >= 0.9)

    model.train()
    return intra_mean, inter_mean, score, class_stats, avg_score, max_score, min_score, median_score, num_classes_great_90

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    samples = sample_windows_capped_npz(
        train_ds,
        clip_len=CLIP_LEN,
        stride=STRIDE,
        windows_per_video=WINDOWS_PER_VIDEO,
        total_cap=TOTAL_WINDOWS_PER_EPOCH,
        seed=SEED + epoch
    )
    print(f"Sampled {len(samples)} windows for this epoch.")

    window_ds = WindowDataset(
        samples, clip_len=CLIP_LEN, retries=RETRIES, jitter=JITTER, pad_mode=PAD_MODE, seed=SEED + epoch
    )

    # train_loader = DataLoader(
    #     window_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=0,               # safe default on shared clusters
    #     worker_init_fn=seed_worker,
    #     generator=g,
    #     collate_fn=safe_collate,
    #     pin_memory=(DEVICE == "cuda"),
    # )
    P, K = 32, 32   # => batch size 256 (independent of BATCH_SIZE var)
    labels_for_sampler = [label_dict[it.cls] for (it, _s) in window_ds.samples]
    pk_sampler = PKBatchSampler(labels_for_sampler, P=P, K=K, drop_last=True)

    train_loader = DataLoader(
        window_ds,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda"),
        batch_sampler=pk_sampler,          # <- use sampler
    )

    test_loader = make_test_loader(
        test_ds, clip_len=CLIP_LEN, stride=STRIDE,
        windows_per_video=2, total_cap=1024, seed=SEED, batch_size=256
    )
    best_score = -1.0

    total_loss, steps = 0.0, 0
    for packed in tqdm(train_loader, desc=f"Train (epoch {epoch+1})", leave=False):
        if packed is None:
            continue
        feats, cls_names, vids = packed

        labels = torch.as_tensor([label_dict[c] for c in cls_names], dtype=torch.long)

        # Augment on CPU, then move once
        shuffled_feats = partial_shuffle_within_window(feats, shuffle_fraction=0.7)
        reverse_feats  = reverse_sequence(feats)
        static_feats   = get_static_window(feats)

        feats          = feats.to(PRIMARY_DEVICE, non_blocking=True)
        shuffled_feats = shuffled_feats.to(PRIMARY_DEVICE, non_blocking=True)
        reverse_feats  = reverse_feats.to(PRIMARY_DEVICE, non_blocking=True)
        static_feats   = static_feats.to(PRIMARY_DEVICE, non_blocking=True)
        labels         = labels.to(PRIMARY_DEVICE, non_blocking=True)

        optimizer.zero_grad()

        emb, _, _   = model(feats)
        sh_emb,_,_  = model(shuffled_feats)
        rev_emb,_,_ = model(reverse_feats)
        st_emb,_,_  = model(static_feats)

        loss_org = loss_fn(emb, labels)
        loss = loss_org + 10.0 * (
            loss_hard(emb, emb, sh_emb) +
            loss_hard(emb, emb, rev_emb) +
            loss_hard(emb, emb, st_emb)
        )

        if not torch.isfinite(loss):
            print("⚠️ Non-finite loss, skipping batch.",
                  f"loss_org={loss_org.item() if torch.isfinite(loss_org) else 'nan'}")
            continue

        loss.backward()
        optimizer.step()
        # cosine_anneal.step()

        total_loss += loss.item()
        # steps += 1
        # if steps % 10 == 0:
        print(f"Step {steps}, Loss: {loss.item():.4f} (Org: {loss_org.item():.4f})")

    # # save model checkpoint
    # os.makedirs("SAVE", exist_ok=True)
    # torch.save(model.state_dict(), f"SAVE/temporal_transformer_epoch{epoch+1:03d}.pt")

    avg_loss = total_loss / max(1, steps)
    intra, inter, score, class_stats, avg_score, max_score, min_score, median_score, num_classes_great_90 = action_consistency(model, test_loader, label_dict, device=PRIMARY_DEVICE)
    print(f"Epoch {epoch+1}: avg train loss {avg_loss:.4f} | avg score {score:.4f} | max score {max_score:.4f} | min score {min_score:.4f} | median score {median_score:.4f} | num classes >= 0.9: {num_classes_great_90} / {len(class_stats)}")
    # # (optional) keep best checkpoint by consistency
    # if np.isfinite(score) and score > best_score:
    #     best_score = score
    #     torch.save(model.state_dict(), f"SAVE/temporal_transformer_best.pt")

print("✅ Training complete.")
