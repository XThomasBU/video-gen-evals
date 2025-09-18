import os
import sys
import cv2
import time
import json
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# make local imports resolvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from human_mesh.TokenHMR.mesh_generator import TokenHMRMeshGenerator
from models import TemporalTransformerV2
from losses import *
from utils import *  # partial_shuffle_within_window, reverse_sequence, get_static_window, etc.

from torch.utils.data import get_worker_info

# Global (per process) worker state
_WORKER = {"device": None, "mesh": None}

def _get_worker_device():
    """Pick a GPU for this worker (or CPU)."""
    info = get_worker_info()
    if DEVICE == "cuda" and NGPUS > 0 and info is not None:
        gpu_id = info.id % NGPUS
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    elif DEVICE == "cuda" and NGPUS > 0:
        # main process fallback (no workers)
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def _get_worker_mesh(mesh_cfg):
    """Lazily create one TokenHMRMeshGenerator per worker, on that worker's GPU."""
    if _WORKER["mesh"] is None:
        dev = _get_worker_device()
        # If TokenHMRMeshGenerator takes a device arg, pass it; otherwise set device via cuda.set_device above.
        cfg = dict(mesh_cfg)
        cfg.setdefault("device", str(dev))  # harmless if unused by the class
        _WORKER["mesh"] = TokenHMRMeshGenerator(config=cfg)
        _WORKER["device"] = dev
    return _WORKER["mesh"]

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


# --------------------------- config ---------------------------

DATASET_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101"
WHITELIST_JSON_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/results_single_ucf101/single"

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

ALL_CLASSES = sorted(os.listdir(DATASET_DIR))
label_dict = {cls: i for i, cls in enumerate(ALL_CLASSES)}

BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 90
CLIP_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== NEW/CHANGED: DataParallel helpers ====
NGPUS = torch.cuda.device_count()
USE_DP = (DEVICE == "cuda") and (NGPUS > 1)
PRIMARY_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")

TOTAL_WINDOWS_PER_EPOCH = 256
WINDOWS_PER_VIDEO = 4      # sampled per video
STRIDE = 1                 # candidate starts spacing


MIN_COVERAGE = 0.80        # accept window if >= 80% frames have meshes
RETRIES = 4                # try up to N nearby starts if coverage too low
JITTER = CLIP_LEN // 2     # +/- range for retry starts (frames)
PAD_MODE = "repeat"        # how to fill missing frames: "repeat" (recommended)

# ------------------------ video dataset -----------------------

class VideoDataset(Dataset):
    def __init__(self, root_dir, items=None, whitelist_json_dir: str = WHITELIST_JSON_DIR):
        self.root_dir = root_dir

        # per-class whitelist
        self.whitelist = {}
        if whitelist_json_dir and os.path.isdir(whitelist_json_dir):
            for fname in sorted(os.listdir(whitelist_json_dir)):
                if not fname.endswith(".json"):
                    continue
                cls_name = os.path.splitext(fname)[0]
                with open(os.path.join(whitelist_json_dir, fname), "r") as f:
                    vids = json.load(f)
                self.whitelist[cls_name] = set(os.path.basename(v) for v in vids)

        if items is None:
            classes_on_disk = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            self.classes = sorted(classes_on_disk)
            self.class_to_videos = {}
            self.all_videos = []
            for cls in self.classes:
                cls_path = os.path.join(root_dir, cls)
                videos = [f for f in os.listdir(cls_path) if f.lower().endswith('.mp4')]
                if self.whitelist:
                    allow = self.whitelist.get(cls, set())
                    videos = [v for v in videos if v in allow]
                if not videos:
                    continue
                videos = sorted(videos)
                self.class_to_videos[cls] = videos
                self.all_videos.extend((cls, v) for v in videos)
        else:
            filt = []
            for cls, v in items:
                if (not self.whitelist) or (v in self.whitelist.get(cls, set())):
                    filt.append((cls, v))
            self.all_videos = filt
            self.classes = sorted(set(cls for cls, _ in filt))
            self.class_to_videos = {cls: sorted([v for c, v in filt if c == cls]) for cls in self.classes}

    def __len__(self): return len(self.all_videos)

    def __getitem__(self, idx):
        cls, video = self.all_videos[idx]
        video_path = os.path.join(self.root_dir, cls, video)
        return cls, video, video_path

def train_test_split(dataset: VideoDataset, train_ratio: float = 0.8, seed: int = 42):
    rng = random.Random(seed)
    train_items, test_items = [], []
    for cls, videos in dataset.class_to_videos.items():
        vids = videos.copy()
        rng.shuffle(vids)
        split_idx = int(len(vids) * train_ratio)
        train_items.extend((cls, v) for v in vids[:split_idx])
        test_items.extend((cls, v) for v in vids[split_idx:])
    return VideoDataset(dataset.root_dir, items=train_items), VideoDataset(dataset.root_dir, items=test_items)

# ------------------- window sampling utils --------------------

def sample_video_windows_capped(video_dataset,
                                clip_len=32,
                                stride=1,
                                windows_per_video=4,
                                total_cap=1000,
                                seed=1337):
    """
    Returns up to `total_cap` (cls, video, path, start, num_frames) tuples.
    Walks videos in random order; for each video picks <= windows_per_video starts.
    """
    rng = random.Random(seed)

    # build a shuffled list of (cls, video)
    all_items = []
    for cls, vids in video_dataset.class_to_videos.items():
        for v in vids:
            all_items.append((cls, v))
    rng.shuffle(all_items)

    samples = []
    for cls, v in all_items:
        if len(samples) >= total_cap:
            break

        path = os.path.join(video_dataset.root_dir, cls, v)
        cap = cv2.VideoCapture(path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        max_start = max(0, num_frames - clip_len)
        if max_start <= 0:
            continue

        # choose up to windows_per_video random starts for THIS video
        possible_starts = list(range(0, max_start + 1, stride))
        k = min(windows_per_video, len(possible_starts),
                total_cap - len(samples))  # don't exceed cap
        chosen = rng.sample(possible_starts, k)
        for s in chosen:
            samples.append((cls, v, path, s, num_frames))
            if len(samples) >= total_cap:
                break

    rng.shuffle(samples)
    return samples

def load_window(video_path, start, length):
    """Decode exactly [start, start+length) frames if available."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(length):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames

# 🔧 NEW: padding utilities for missing/invalid frames -----------------

def _nearest_fill_indices(valid_idx, T):
    """
    Build a mapping for each t in [0..T-1] to nearest valid index.
    """
    valid = np.array(sorted(valid_idx))
    out = np.empty(T, dtype=int)
    for t in range(T):
        j = np.abs(valid - t).argmin()
        out[t] = valid[j]
    return out

def pad_or_reject_mesh_seq(mesh_seq, T, min_coverage=0.8, pad_mode="repeat"):
    """
    mesh_seq: dict {frame_offset:int -> dict(params)}
      where frame_offset is 0..T-1 relative to the window start.
    Returns:
      ordered list of length T (each is params dict) or None to reject.
    """
    valid_idx = [k for k in mesh_seq.keys() if 0 <= k < T]
    if len(valid_idx) == 0:
        return None
    coverage = len(valid_idx) / float(T)
    if coverage < min_coverage:
        return None

    ordered = [None] * T
    # place knowns
    for k in valid_idx:
        ordered[k] = mesh_seq[k]

    if pad_mode == "repeat":
        # repeat nearest valid frame for missing positions
        nearest = _nearest_fill_indices(valid_idx, T)
        for t in range(T):
            if ordered[t] is None:
                ordered[t] = mesh_seq[nearest[t]]
        return ordered

    # fallback: reject if unknown pad_mode
    return None

# --------------------- window dataset -------------------------

class WindowDataset(Dataset):
    """
    Loads only the requested windows; retries nearby starts if mesh coverage too low;
    pads missing frames by repeating nearest valid mesh.
    """
    def __init__(self, samples, clip_len=32, mesh_generator=None,
                 min_coverage=0.8, retries=4, jitter=16, pad_mode="repeat", seed=1337):
        self.samples = samples  # list of (cls, video, path, start, num_frames)
        self.clip_len = clip_len
        self.mesh_generator = mesh_generator
        self.min_cov = min_coverage
        self.retries = retries
        self.jitter = jitter
        self.pad_mode = pad_mode
        self.rng = random.Random(seed)

    def __len__(self): return len(self.samples)

    def _try_one(self, cls, video, path, start):
        # decode only the window
        frames = load_window(path, start, self.clip_len)
        if len(frames) == 0:
            return None
        mesh_data = self.mesh_generator.process_video(frames)
        if mesh_data is False or mesh_data is None:
            return None

        # mesh_data is assumed {frame_idx_within_window:int -> params dict}
        # normalize keys to 0..T-1 if needed
        # if mesh_data keys are absolute, convert to relative by sorting and reindexing:
        # Here we assume keys are 0..len(frames)-1 (common for per-call processing).
        # If your mesh_generator returns arbitrary keys, adapt below.
        rel_map = {}
        for k, v in mesh_data.items():
            rel_k = int(k)
            if 0 <= rel_k < self.clip_len:
                rel_map[rel_k] = v

        ordered = pad_or_reject_mesh_seq(rel_map, self.clip_len, self.min_cov, self.pad_mode)
        if ordered is None:
            return None

        # stack tensors consistently
        pose  = torch.stack([torch.tensor(f['pose']) for f in ordered], dim=0)      # [T, 207]
        betas = torch.stack([torch.tensor(f['betas']) for f in ordered], dim=0)          # [T, 10]
        glob  = torch.stack([torch.tensor(f['global_orient']) for f in ordered], dim=0)   # [T, 9]
        vit   = torch.stack([torch.tensor(f['vit']) for f in ordered], dim=0)             # [T, 1024]
        # kp2d  = torch.stack([torch.tensor(f['twod_kp']) for f in ordered], dim=0)        # [T, 120]
        # dino_vit = torch.stack([torch.tensor(f['dino_vit']) for f in ordered], dim=0)     # [T, 1024]

        # motion
        d_vit = _vit_delta(vit)
        d_go = _rot_matrix_delta(glob.reshape(32, -1))
        d_pose = _rot_axisangle_delta(pose.reshape(32, -1))
        d_bet = _betas_delta(betas)

        feats = np.concatenate([
            pose.reshape(self.clip_len, -1),
            betas,
            glob.reshape(self.clip_len, -1),
            vit
        ], axis=-1)

        motion_feats = np.concatenate([
            d_pose.reshape(self.clip_len, -1),
            d_bet,
            d_go.reshape(self.clip_len, -1),
            d_vit
        ], axis=-1)

        enriched = np.concatenate([feats, motion_feats], axis=-1)

        return torch.tensor(enriched, dtype=torch.float32), cls, video

    def __getitem__(self, idx):
        cls, video, path, start, num_frames = self.samples[idx]

        # first attempt
        out = self._try_one(cls, video, path, start)
        if out is not None:
            return out

        # 🔧 NEW: retries with random jitter in [-JITTER, +JITTER]
        for _ in range(self.retries):
            # sample a new start near original, clamp to valid range
            delta = self.rng.randint(-self.jitter, self.jitter)
            new_start = max(0, min(start + delta, max(0, num_frames - self.clip_len)))
            out = self._try_one(cls, video, path, new_start)
            if out is not None:
                return out

        # give up; signal failure for collate_fn to drop
        return None

# 🔧 NEW: collate that filters failed samples -------------------

def safe_collate(batch):
    """Drop Nones produced by WindowDataset when a window couldn't be repaired."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # return an empty batch that the train loop can skip
        return None
    feats, cls_names, vids = zip(*batch)
    feats = torch.stack(feats, dim=0)
    return feats, list(cls_names), list(vids)

# ------------------------- training ---------------------------


full_dataset = VideoDataset(DATASET_DIR)
train_ds, test_ds = train_test_split(full_dataset, train_ratio=0.8, seed=SEED)
print(f"Train videos: {len(train_ds)}, Test videos: {len(test_ds)}")

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": False, "full_frame": True}
)
print(mesh_generator)

# ==== NEW/CHANGED: create modules on the primary device first
model = TemporalTransformerV2(input_dim=1250*2, latent_dim=LATENT_DIM).to(PRIMARY_DEVICE)
arc = ArcMarginProduct(LATENT_DIM, len(ALL_CLASSES), s=30.0, m=0.35).to(PRIMARY_DEVICE)

# Optionally wrap in DataParallel
if USE_DP:
    print(f"Using DataParallel across {NGPUS} GPUs: {list(range(NGPUS))}")
    model = torch.nn.DataParallel(model, device_ids=list(range(NGPUS)))
    arc = torch.nn.DataParallel(arc, device_ids=list(range(NGPUS)))

loss_fn = TCL().to(PRIMARY_DEVICE)
loss_hard = SupConWithHardNegatives().to(PRIMARY_DEVICE)

# IMPORTANT: build optimizer *after* (possible) DP wrapping so params are correct
params = list(model.parameters()) + list(arc.parameters())
optimizer = torch.optim.AdamW(params, lr=3e-4)
cosine_anneal = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_ds) * EPOCHS, eta_min=1e-6
)

from tqdm import tqdm

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # NEW: capped sampling
    samples = sample_video_windows_capped(
        train_ds,
        clip_len=CLIP_LEN,
        stride=STRIDE,
        windows_per_video=WINDOWS_PER_VIDEO,
        total_cap=TOTAL_WINDOWS_PER_EPOCH,
        seed=SEED + epoch
    )
    print(f"Sampled {len(samples)} windows for this epoch.")

    window_ds = WindowDataset(
        samples, clip_len=CLIP_LEN, mesh_generator=mesh_generator,
        min_coverage=MIN_COVERAGE, retries=RETRIES, jitter=JITTER,
        pad_mode=PAD_MODE, seed=SEED + epoch
    )

    train_loader = DataLoader(
        window_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda"),     # <-- NEW
    )

    total_loss, steps = 0.0, 0
    for packed in tqdm(train_loader, desc=f"Train (epoch {epoch+1})", leave=False):
        if packed is None:
            continue
        feats, cls_names, vids = packed

        # ==== NEW/CHANGED: build labels on CPU, then move once
        labels = torch.as_tensor([label_dict[c] for c in cls_names], dtype=torch.long)

        # Create augmented views on CPU (as you already do), then push once
        shuffled_feats = partial_shuffle_within_window(feats, shuffle_fraction=0.7)
        reverse_feats  = reverse_sequence(feats)
        static_feats   = get_static_window(feats)

        # (Optional) keep this print; shapes are CPU-side so it's cheap
        print(feats.shape, shuffled_feats.shape, reverse_feats.shape, static_feats.shape)

        # ==== NEW/CHANGED: non_blocking moves to the primary device
        feats          = feats.to(PRIMARY_DEVICE, non_blocking=True)
        shuffled_feats = shuffled_feats.to(PRIMARY_DEVICE, non_blocking=True)
        reverse_feats  = reverse_feats.to(PRIMARY_DEVICE, non_blocking=True)
        static_feats   = static_feats.to(PRIMARY_DEVICE, non_blocking=True)
        labels         = labels.to(PRIMARY_DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # After this point, DataParallel (if enabled) will split the first tensor dimension
        embeddings, _, _ = model(feats)
        sh_emb,   _, _   = model(shuffled_feats)
        rev_emb,  _, _   = model(reverse_feats)
        st_emb,   _, _   = model(static_feats)

        loss_org = loss_fn(embeddings, labels)
        loss = loss_org + 10.0 * (
            loss_hard(embeddings, embeddings, sh_emb) +
            loss_hard(embeddings, embeddings, rev_emb) +
            loss_hard(embeddings, embeddings, st_emb)
        )
        if not torch.isfinite(loss):
            print("⚠️  Non-finite loss, skipping batch.",
                f"loss_org={loss_org.item() if torch.isfinite(loss_org) else 'nan'}")
            continue

        loss.backward()
        optimizer.step()
        cosine_anneal.step()

        total_loss += loss.item()
        print(f"Step {steps+1}, Loss: {loss.item():.4f} (Org: {loss_org.item():.4f})")
        steps += 1

    # save model
    torch.save(model.state_dict(), f"SAVE/temporal_transformer_epoch{epoch+1:03d}.pt")