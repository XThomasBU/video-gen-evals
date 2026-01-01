import torch
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import Sampler, BatchSampler
from torch.utils.data import Dataset, DataLoader
import typing as T
from tqdm import tqdm
import random
import os
import json
import datetime
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import math
# import PCA
from sklearn.decomposition import PCA

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


def split_modalities(seqs, dim_map_raw, dim_map_diff):
    """
    Given [B, T, D], return:
    - dict of {modality: [B, T, D_raw]}
    - dict of {modality: [B, T, D_diff]}
    """
    raw_feats = {}
    diff_feats = {}

    # Split raw
    start = 0
    for mod, d in dim_map_raw.items():
        raw_feats[mod] = seqs[:, :, start:start + d]
        start += d

    # Split diff
    for mod, d in dim_map_diff.items():
        diff_feats[mod] = seqs[:, :, start:start + d]
        start += d

    return raw_feats, diff_feats


def merge_modalities(raw_feats, diff_feats, dim_map_raw, dim_map_diff):
    """
    Return [B, T, D] concatenated back.
    """
    raw = [raw_feats[m] for m in dim_map_raw]
    diff = [diff_feats[m] for m in dim_map_diff]
    return torch.cat(raw + diff, dim=-1)


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

def collate_fn(batch):
    sequences, labels, vid_ids, window_ids = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    sequences = pad_sequence(sequences, batch_first=True)  # [B, T_max, D]
    labels = torch.tensor(labels)
    return sequences, lengths, labels, vid_ids, window_ids

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    feats, cls_names, vids = zip(*batch)
    feats = torch.stack(feats, dim=0)
    return feats, list(cls_names), list(vids)

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

def _rotmat_delta(R: torch.Tensor) -> torch.Tensor:
    """
    R: [T, J, 3, 3] or [T, 3, 3]
    returns axis-angle deltas with same leading dims but last dim = 3 (per joint)
    """
    T = R.shape[0]
    R_prev = torch.cat([R[:1], R[:-1]], dim=0)
    Rrel = torch.matmul(R_prev.transpose(-1, -2), R)  # [...,3,3]
    w = _log_so3(Rrel)  # [...,3]
    return w


def _procrustes_kp_delta(kp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Procrustes (translation + scale + rotation) normalized keypoint velocity.
    kp: [T, 2*K] or [T, K, 2] with x,y in [0,1]
    returns: [T, 2*K]
    """
    if kp.dim() == 3:        # [T, K, 2]
        T, K, _ = kp.shape
        pts = kp
    else:                    # [T, 2*K]
        T, D = kp.shape
        K = D // 2
        pts = kp.view(T, K, 2)

    # 1) remove translation (center each frame)
    pts_c = pts - pts.mean(dim=1, keepdim=True)                       # [T,K,2]

    # 2) remove scale
    s = torch.linalg.norm(pts_c, dim=(1, 2), keepdim=True).clamp_min(eps)  # [T,1,1]
    pts_n = pts_c / s                                                      # [T,K,2]

    # 3) align rotations across consecutive frames
    deltas = torch.zeros_like(pts_n)
    deltas[0] = 0.0

    for t in range(1, T):
        X = pts_n[t-1]                 # [K,2]
        Y = pts_n[t]                   # [K,2]

        # Kabsch: find rotation R that best aligns X -> Y
        H = X.t().matmul(Y)            # [2,2]
        U, _, Vh = torch.linalg.svd(H)
        R = Vh @ U.t()
        if torch.det(R) < 0:
            Vh[:, -1] *= -1
            R = Vh @ U.t()

        X_aligned = X @ R              # [K,2]
        deltas[t] = Y - X_aligned

    return deltas.reshape(T, K * 2)

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
                 whitelist_json_dir: T.Optional[str] = None, filter_classes: T.Optional[T.List[str]] = None, min_videos_per_class: int = 10, enforce_min_per_class: bool = False):
        self.root_dir = root_dir
        self.whitelist = self._load_whitelist(whitelist_json_dir) if whitelist_json_dir else {}
        self.filter_classes = filter_classes

        raw_items = items if items is not None else self._scan()

        # group items by class
        class_to_items: T.Dict[str, T.List[VideoItem]] = {}
        for it in raw_items:
            class_to_items.setdefault(it.cls, []).append(it)

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
        for cls in tqdm(sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]), desc="Scanning dataset", total=len(os.listdir(self.root_dir))):
            if self.filter_classes and (cls in self.filter_classes):
                cls_dir = os.path.join(self.root_dir, cls)
                count = 0
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
                        # count += 1
                        # if count > 10:
                        #     break  # limit to first 10 videos per class for speed
                    except Exception:
                        # Skip corrupted entries silently; you can log if desired
                        print(f"Failed for {f}")
                        continue
            if self.filter_classes is None:
                cls_dir = os.path.join(self.root_dir, cls)
                count = 0
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
                        # count += 1
                        # if count > 10:
                        #     break  # limit to first 10 videos per class for speed
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

# ----------------------- Window dataset ------------------------

class WindowDataset(Dataset):
    """
    Loads windows from .npz arrays and constructs features + motion deltas.
    """
    def __init__(self, samples: T.List[T.Tuple[VideoItem,int]],
                 clip_len: int = 32,
                 seed: int = 1337,
                 keypoint_dir: T.Optional[str] = None,
                 clip_dir=None,
                 dino_dir=None,
                 stats: T.Optional['ModalityStats'] = None):
        self.samples = samples
        self.clip_len = clip_len
        self.rng = random.Random(seed)
        self.stats = stats
        self.keypoint_dir = keypoint_dir
        self.clip_dir = clip_dir
        self.dino_dir = dino_dir

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
        npz = np.load(it.path, mmap_mode="r")
        pose = npz["pose"]
        betas = npz["betas"]
        gori = npz["global_orient"]
        vit = npz["vit"]

        pose_w  = self._slice_or_pad(pose,  start, self.clip_len)
        betas_w = self._slice_or_pad(betas, start, self.clip_len)
        gori_w  = self._slice_or_pad(gori,  start, self.clip_len)
        vit_w   = self._slice_or_pad(vit,   start, self.clip_len)
        T = pose_w.shape[0]

        pose_R = torch.from_numpy(pose_w).float()
        gori_R = torch.from_numpy(gori_w).float()
        betas_t = torch.from_numpy(betas_w).float()
        vit_t   = torch.from_numpy(vit_w).float()

        pose_raw  = pose_R.reshape(T, -1)
        gori_raw  = gori_R.reshape(T, -1)
        vit_raw   = vit_t
        beta_raw  = betas_t

        cls_name = it.cls
        vid_stem = os.path.splitext(os.path.basename(it.path))[0]
        
        keypoints_raw = None
        if self.keypoint_dir is not None:
            if "SAVE_GEN" in self.keypoint_dir or "SAVE_NEW" in self.keypoint_dir or "generated_kps" in self.keypoint_dir:
                kp_path = os.path.join(self.keypoint_dir, vid_stem, "keypoints.npy")
            else:
                kp_path = os.path.join(self.keypoint_dir, cls_name, vid_stem, "keypoints.npy")

            if not os.path.exists(kp_path):
                raise FileNotFoundError(f"Expected keypoints at '{kp_path}' for video '{vid_stem}' but file does not exist.")

            try:
                kp_np = np.load(kp_path)
                kp_w = self._slice_or_pad(kp_np, start, self.clip_len)
                keypoints_raw = torch.from_numpy(kp_w).float()
            except Exception as e:
                raise RuntimeError(f"Failed to load keypoints from '{kp_path}' for video '{vid_stem}': {e}")

        clip_raw = None
        if self.clip_dir is not None:
            try:
                if "SAVE_GEN" in str(self.keypoint_dir) or "SAVE_NEW" in str(self.keypoint_dir) or "generated_kps" in str(self.keypoint_dir):
                    clip_path = os.path.join(self.clip_dir, vid_stem, "clip_embeddings.npz")
                else:
                    clip_path = os.path.join(self.clip_dir, cls_name, vid_stem, "clip_embeddings.npz")
                if os.path.exists(clip_path):
                    clip_np = np.load(clip_path)["embeddings"]
                    clip_w = self._slice_or_pad(clip_np, start, self.clip_len)
                    clip_raw = torch.from_numpy(clip_w).float()
            except Exception:
                pass

        dino_raw = None
        if self.dino_dir is not None:
            try:
                if "SAVE_GEN" in str(self.keypoint_dir) or "SAVE_NEW" in str(self.keypoint_dir) or "generated_kps" in str(self.keypoint_dir):
                    dino_path = os.path.join(self.dino_dir, vid_stem, "dino_embeddings.npz")
                else:
                    dino_path = os.path.join(self.dino_dir, cls_name, vid_stem, "dino_embeddings.npz")
                if os.path.exists(dino_path):
                    dino_np = np.load(dino_path)["embeddings"]
                    dino_w = self._slice_or_pad(dino_np, start, self.clip_len)
                    dino_raw = torch.from_numpy(dino_w).float()
            except Exception:
                pass

        # ----- Motion (diff) features -----
        vit_diff = _vit_delta(vit_t)
        pose_diff = _rotmat_delta(pose_R).reshape(T, -1)
        gori_diff = _rotmat_delta(gori_R).reshape(T, -1)
        beta_diff = _betas_delta(beta_raw)
        
        keypoints_diff = None
        if keypoints_raw is not None:
            keypoints_diff = _procrustes_kp_delta(keypoints_raw)
        
        clip_diff = None
        if clip_raw is not None:
            clip_diff = _vit_delta(clip_raw)
        
        dino_diff = None
        if dino_raw is not None:
            dino_diff = _vit_delta(dino_raw)

        if self.stats is not None:
            eps = 1e-6
            vit_raw  = (vit_raw  - self.stats.vit_raw_mean)  / (self.stats.vit_raw_std  + eps)
            gori_raw = (gori_raw - self.stats.gori_raw_mean) / (self.stats.gori_raw_std + eps)
            pose_raw = (pose_raw - self.stats.pose_raw_mean) / (self.stats.pose_raw_std + eps)
            beta_raw = (beta_raw - self.stats.beta_raw_mean) / (self.stats.beta_raw_std + eps)
            if keypoints_raw is not None:
                keypoints_raw = (keypoints_raw - self.stats.keypoints_raw_mean) / (self.stats.keypoints_raw_std + eps)
            if clip_raw is not None:
                clip_raw = (clip_raw - self.stats.clip_raw_mean) / (self.stats.clip_raw_std + eps)
            if dino_raw is not None:
                dino_raw = (dino_raw - self.stats.dino_raw_mean) / (self.stats.dino_raw_std + eps)

            vit_diff  = (vit_diff  - self.stats.vit_diff_mean)  / (self.stats.vit_diff_std  + eps)
            gori_diff = (gori_diff - self.stats.gori_diff_mean) / (self.stats.gori_diff_std + eps)
            pose_diff = (pose_diff - self.stats.pose_diff_mean) / (self.stats.pose_diff_std + eps)
            beta_diff = (beta_diff - self.stats.beta_diff_mean) / (self.stats.beta_diff_std + eps)
            if keypoints_diff is not None:
                keypoints_diff = (keypoints_diff - self.stats.keypoints_diff_mean) / (self.stats.keypoints_diff_std + eps)
            if clip_diff is not None:
                clip_diff = (clip_diff - self.stats.clip_diff_mean) / (self.stats.clip_diff_std + eps)
            if dino_diff is not None:
                dino_diff = (dino_diff - self.stats.dino_diff_mean) / (self.stats.dino_diff_std + eps)

        raw_parts = [vit_raw, gori_raw, pose_raw, beta_raw]
        diff_parts = [vit_diff, gori_diff, pose_diff, beta_diff]
        
        if keypoints_raw is not None:
            raw_parts.append(keypoints_raw)
        if keypoints_diff is not None:
            diff_parts.append(keypoints_diff)
        if clip_raw is not None:
            raw_parts.append(clip_raw)
        if clip_diff is not None:
            diff_parts.append(clip_diff)
        if dino_raw is not None:
            raw_parts.append(dino_raw)
        if dino_diff is not None:
            diff_parts.append(dino_diff)

        raw  = torch.cat(raw_parts, dim=-1)
        diff = torch.cat(diff_parts, dim=-1)
        feats = torch.cat([raw, diff], dim=-1)

        return feats, it.cls, it.name

    def __getitem__(self, idx):
        it, start = self.samples[idx]
        out = self._try_one(it, start)
        if out is not None:
            return out
        return None



class SequenceDataset(WindowDataset):
    """
    Groups all windows of a video into one stacked tensor.
    Each __getitem__ gives:
        windows: [N_windows, clip_len, D_feat]
        cls:     str
        name:    str
    """
    def __init__(self, videos: T.List[VideoItem],
                 clip_len: int = 32,
                 stride: int = 8,
                 seed: int = 1337,
                 keypoint_dir: T.Optional[str] = None,
                 stats: T.Optional['ModalityStats'] = None):
        # Instead of samples (video,start), we just keep videos
        self.videos = videos
        self.clip_len = clip_len
        self.stride = stride
        super().__init__([], clip_len, seed, keypoint_dir, stats)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        it = self.videos[idx]
        # load video length
        npz = np.load(it.path, mmap_mode="r")
        num_frames = npz["vit"].shape[0]

        windows = []
        for start in range(0, max(1, num_frames - self.clip_len + 1), self.stride):
            feats, _, _ = self._try_one(it, start)   # reuse WindowDataset logic
            windows.append(feats)

        if not windows:
            return None

        windows = torch.stack(windows, dim=0)  # [N_windows, clip_len, D_feat]
        return windows, it.cls, it.name


from dataclasses import asdict, dataclass

@dataclass
class ModalityStats:
    vit_raw_mean:  torch.Tensor; vit_raw_std:  torch.Tensor
    gori_raw_mean: torch.Tensor; gori_raw_std: torch.Tensor
    pose_raw_mean: torch.Tensor; pose_raw_std: torch.Tensor
    beta_raw_mean: torch.Tensor; beta_raw_std: torch.Tensor
    keypoints_raw_mean: torch.Tensor; keypoints_raw_std: torch.Tensor
    clip_raw_mean: torch.Tensor; clip_raw_std: torch.Tensor
    dino_raw_mean: torch.Tensor; dino_raw_std: torch.Tensor

    vit_diff_mean:  torch.Tensor; vit_diff_std:  torch.Tensor
    gori_diff_mean: torch.Tensor; gori_diff_std: torch.Tensor
    pose_diff_mean: torch.Tensor; pose_diff_std: torch.Tensor
    beta_diff_mean: torch.Tensor; beta_diff_std: torch.Tensor
    keypoints_diff_mean: torch.Tensor; keypoints_diff_std: torch.Tensor
    clip_diff_mean: torch.Tensor; clip_diff_std: torch.Tensor
    dino_diff_mean: torch.Tensor; dino_diff_std: torch.Tensor


def _update_sum_sum2(X: np.ndarray, sum1: np.ndarray, sum2: np.ndarray):
    # X: [T, D]
    sum1 += X.sum(axis=0, dtype=np.float64)
    sum2 += (X.astype(np.float64) ** 2).sum(axis=0)
    return sum1, sum2, X.shape[0]

def compute_stats_from_npz(train_items: T.List[VideoItem], keypoint_dir: str, clip_dir: T.Optional[str] = None, dino_dir: T.Optional[str] = None, eps: float = 1e-6) -> ModalityStats:
    """
    Stream over TRAIN .npz files and compute per-dim mean/std for:
      vit_raw, gori_raw, pose_raw, beta_raw, and their DIFF counterparts.

    Conventions (matching WindowDataset):
      - vit_diff: L2-normalize per-frame then v - v_prev (first row self-diff)
      - gori_diff: relative rotation from 3x3 matrices -> axis-angle (3 dims)
      - pose_diff: relative rotation per joint from 3x3 matrices -> axis-angle (3*J dims)
      - beta_diff: finite difference (first row self-diff)
    """
    assert len(train_items) > 0, "compute_stats_from_npz: train_items is empty"

    # Infer dims once
    npz0 = np.load(train_items[0].path, mmap_mode="r")
    vitD = int(npz0["vit"].shape[1])
    J    = int(npz0["pose"].shape[1])  # number of joints

    # RAW dims (flattened rotation matrices)
    D_vit_raw  = vitD
    D_gori_raw = 9
    D_pose_raw = 9 * J
    D_beta_raw = 10
    D_kp_raw   = 120  # assuming 60 keypoints with (x,y)
    D_clip_raw = 512
    D_dino_raw = 768

    # DIFF dims (axis-angle per joint / orient)
    D_vit_diff  = vitD
    D_gori_diff = 3
    D_pose_diff = 3 * J
    D_beta_diff = 10
    D_kp_diff   = 120  # assuming 60 keypoints with (x,y)
    D_clip_diff = 512
    D_dino_diff = 768

    def zeros(D): return np.zeros((D,), dtype=np.float64)

    # Allocate accumulators (float64 for stability)
    s_vit_raw,  ss_vit_raw,  n_vit_raw  = zeros(D_vit_raw),  zeros(D_vit_raw),  0
    s_gori_raw, ss_gori_raw, n_gori_raw = zeros(D_gori_raw), zeros(D_gori_raw), 0
    s_pose_raw, ss_pose_raw, n_pose_raw = zeros(D_pose_raw), zeros(D_pose_raw), 0
    s_beta_raw, ss_beta_raw, n_beta_raw = zeros(D_beta_raw), zeros(D_beta_raw), 0
    s_kp_raw,   ss_kp_raw,   n_kp_raw   = zeros(D_kp_raw),   zeros(D_kp_raw),   0
    s_clip_raw,  ss_clip_raw,  n_clip_raw  = zeros(D_clip_raw),  zeros(D_clip_raw),  0
    s_dino_raw,  ss_dino_raw,  n_dino_raw  = zeros(D_dino_raw),  zeros(D_dino_raw),  0

    s_vit_diff,  ss_vit_diff,  n_vit_diff  = zeros(D_vit_diff),  zeros(D_vit_diff),  0
    s_gori_diff, ss_gori_diff, n_gori_diff = zeros(D_gori_diff), zeros(D_gori_diff), 0
    s_pose_diff, ss_pose_diff, n_pose_diff = zeros(D_pose_diff), zeros(D_pose_diff), 0
    s_beta_diff, ss_beta_diff, n_beta_diff = zeros(D_beta_diff), zeros(D_beta_diff), 0
    s_kp_diff,   ss_kp_diff,   n_kp_diff   = zeros(D_kp_diff),   zeros(D_kp_diff),   0
    s_clip_diff, ss_clip_diff,  n_clip_diff  = zeros(D_clip_diff),  zeros(D_clip_diff),  0
    s_dino_diff, ss_dino_diff,  n_dino_diff  = zeros(D_dino_diff),  zeros(D_dino_diff),  0

    for it in tqdm(train_items, desc="Computing stats from .npz", total=len(train_items)):
        npz = np.load(it.path, mmap_mode="r")
        pose  = npz["pose"].astype(np.float32)           # [T,J,3,3]
        gori  = npz["global_orient"].astype(np.float32)  # [T,3,3]
        betas = npz["betas"].astype(np.float32)          # [T,10]
        vit   = npz["vit"].astype(np.float32)            # [T,D]
        Tlen  = pose.shape[0]

        # ---------- RAW (state) ----------
        pose_raw = pose.reshape(Tlen, -1)   # [T, 9*J]
        gori_raw = gori.reshape(Tlen, -1)   # [T, 9]
        vit_raw  = vit                      # [T, D]
        beta_raw = betas                    # [T, 10]

        cls_name = it.cls
        vid_stem = os.path.splitext(os.path.basename(it.path))[0]
        
        keypoints_raw = None
        if keypoint_dir is not None:
            try:
                if "SAVE_GEN" in keypoint_dir or "SAVE_NEW" in keypoint_dir or "generated_kps" in keypoint_dir:
                    kp_path = os.path.join(keypoint_dir, vid_stem, "keypoints.npy")
                else:
                    kp_path = os.path.join(keypoint_dir, cls_name, vid_stem, "keypoints.npy")
                if os.path.exists(kp_path):
                    kp_np = np.load(kp_path)
                    keypoints_raw = kp_np
            except Exception:
                pass

        clip_raw = None
        if clip_dir is not None:
            try:
                if "SAVE_GEN" in str(keypoint_dir) or "SAVE_NEW" in str(keypoint_dir) or "generated_kps" in str(keypoint_dir):
                    clip_path = os.path.join(clip_dir, vid_stem, "clip_embeddings.npz")
                else:
                    clip_path = os.path.join(clip_dir, cls_name, vid_stem, "clip_embeddings.npz")
                if os.path.exists(clip_path):
                    clip_np = np.load(clip_path)["embeddings"]
                    clip_raw = clip_np
            except Exception:
                pass

        dino_raw = None
        if dino_dir is not None:
            try:
                if "SAVE_GEN" in str(keypoint_dir) or "SAVE_NEW" in str(keypoint_dir) or "generated_kps" in str(keypoint_dir):
                    dino_path = os.path.join(dino_dir, vid_stem, "dino_embeddings.npz")
                else:
                    dino_path = os.path.join(dino_dir, cls_name, vid_stem, "dino_embeddings.npz")
                if os.path.exists(dino_path):
                    dino_np = np.load(dino_path)["embeddings"]
                    dino_raw = dino_np
            except Exception:
                pass

        s_vit_raw,  ss_vit_raw,  c = _update_sum_sum2(vit_raw,  s_vit_raw,  ss_vit_raw);  n_vit_raw  += c
        s_gori_raw, ss_gori_raw, c = _update_sum_sum2(gori_raw, s_gori_raw, ss_gori_raw); n_gori_raw += c
        s_pose_raw, ss_pose_raw, c = _update_sum_sum2(pose_raw, s_pose_raw, ss_pose_raw); n_pose_raw += c
        s_beta_raw, ss_beta_raw, c = _update_sum_sum2(beta_raw, s_beta_raw, ss_beta_raw); n_beta_raw += c
        if keypoints_raw is not None:
            s_kp_raw,  ss_kp_raw,   c = _update_sum_sum2(keypoints_raw,   s_kp_raw,   ss_kp_raw);   n_kp_raw   += c
        if clip_raw is not None:
            s_clip_raw,  ss_clip_raw,  c = _update_sum_sum2(clip_raw,  s_clip_raw,  ss_clip_raw);  n_clip_raw  += c
        if dino_raw is not None:
            s_dino_raw,  ss_dino_raw,  c = _update_sum_sum2(dino_raw,  s_dino_raw,  ss_dino_raw);  n_dino_raw  += c

        vit_diff  = _vit_delta(torch.from_numpy(vit_raw).float()).numpy()
        pose_diff = _rotmat_delta(torch.from_numpy(pose).float()).reshape(Tlen, -1).numpy()
        gori_diff = _rotmat_delta(torch.from_numpy(gori).float()).reshape(Tlen, -1).numpy()
        beta_diff = _betas_delta(torch.from_numpy(beta_raw).float()).numpy()
        
        keypoints_diff = None
        if keypoints_raw is not None:
            keypoints_diff = _procrustes_kp_delta(torch.from_numpy(keypoints_raw).float()).numpy()
        
        clip_diff = None
        if clip_raw is not None:
            clip_diff = _vit_delta(torch.from_numpy(clip_raw).float()).numpy()
        
        dino_diff = None
        if dino_raw is not None:
            dino_diff = _vit_delta(torch.from_numpy(dino_raw).float()).numpy()

        s_vit_diff,  ss_vit_diff,  c = _update_sum_sum2(vit_diff,  s_vit_diff,  ss_vit_diff);  n_vit_diff  += c
        s_gori_diff, ss_gori_diff, c = _update_sum_sum2(gori_diff, s_gori_diff, ss_gori_diff); n_gori_diff += c
        s_pose_diff, ss_pose_diff, c = _update_sum_sum2(pose_diff, s_pose_diff, ss_pose_diff); n_pose_diff += c
        s_beta_diff, ss_beta_diff, c = _update_sum_sum2(beta_diff, s_beta_diff, ss_beta_diff); n_beta_diff += c
        if keypoints_diff is not None:
            s_kp_diff,  ss_kp_diff,   c = _update_sum_sum2(keypoints_diff,   s_kp_diff,   ss_kp_diff);   n_kp_diff   += c
        if clip_diff is not None:
            s_clip_diff,  ss_clip_diff,  c = _update_sum_sum2(clip_diff,  s_clip_diff,  ss_clip_diff);  n_clip_diff  += c
        if dino_diff is not None:
            s_dino_diff,  ss_dino_diff,  c = _update_sum_sum2(dino_diff,  s_dino_diff,  ss_dino_diff);  n_dino_diff  += c

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
    
    keypoints_raw_mean, keypoints_raw_std = (None, None)
    if n_kp_raw > 0:
        keypoints_raw_mean, keypoints_raw_std = finalize(s_kp_raw, ss_kp_raw, n_kp_raw)
    
    clip_raw_mean, clip_raw_std = (None, None)
    if clip_dir is not None and n_clip_raw > 0:
        clip_raw_mean, clip_raw_std = finalize(s_clip_raw, ss_clip_raw, n_clip_raw)
    
    dino_raw_mean, dino_raw_std = (None, None)
    if dino_dir is not None and n_dino_raw > 0:
        dino_raw_mean, dino_raw_std = finalize(s_dino_raw, ss_dino_raw, n_dino_raw)

    vit_diff_mean,  vit_diff_std  = finalize(s_vit_diff,  ss_vit_diff,  n_vit_diff)
    gori_diff_mean, gori_diff_std = finalize(s_gori_diff, ss_gori_diff, n_gori_diff)
    pose_diff_mean, pose_diff_std = finalize(s_pose_diff, ss_pose_diff, n_pose_diff)
    beta_diff_mean, beta_diff_std = finalize(s_beta_diff, ss_beta_diff, n_beta_diff)
    
    keypoints_diff_mean, keypoints_diff_std = (None, None)
    if n_kp_diff > 0:
        keypoints_diff_mean, keypoints_diff_std = finalize(s_kp_diff, ss_kp_diff, n_kp_diff)
    
    clip_diff_mean, clip_diff_std = (None, None)
    if clip_dir is not None and n_clip_diff > 0:
        clip_diff_mean, clip_diff_std = finalize(s_clip_diff, ss_clip_diff, n_clip_diff)
    
    dino_diff_mean, dino_diff_std = (None, None)
    if dino_dir is not None and n_dino_diff > 0:
        dino_diff_mean, dino_diff_std = finalize(s_dino_diff, ss_dino_diff, n_dino_diff)

    return ModalityStats(
        vit_raw_mean, vit_raw_std,
        gori_raw_mean, gori_raw_std,
        pose_raw_mean, pose_raw_std,
        beta_raw_mean, beta_raw_std,
        keypoints_raw_mean, keypoints_raw_std,
        clip_raw_mean, clip_raw_std,
        dino_raw_mean, dino_raw_std,
        vit_diff_mean, vit_diff_std,
        gori_diff_mean, gori_diff_std,
        pose_diff_mean, pose_diff_std,
        beta_diff_mean, beta_diff_std,
        keypoints_diff_mean, keypoints_diff_std,
        clip_diff_mean, clip_diff_std,
        dino_diff_mean, dino_diff_std,
    )

def make_test_loader(
    ds: NpzVideoDataset,
    clip_len: int,
    stride: int,
    keypoint_dir: str,
    clip_dir: T.Optional[str] = None,
    dino_dir: T.Optional[str] = None,
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
        seed=seed,
        stats=stats,
        keypoint_dir=keypoint_dir,
        clip_dir=clip_dir,
        dino_dir=dino_dir,
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

# def sample_all_windows_npz(
#     ds: "NpzVideoDataset",
#     clip_len: int = 32,
#     stride:   int = 8,
# ):
#     """
#     Return *all* valid (VideoItem, start) windows from the dataset.
#     No random sampling, no caps — full coverage.
#     """
#     out: T.List[T.Tuple["VideoItem", int]] = []

#     for it in tqdm(ds.items, desc="Enumerating all windows", total=len(ds.items)):
#         max_start = it.length - clip_len
#         if max_start < 0:
#             continue
#         starts = range(0, max_start + 1, stride)
#         for s in starts:
#             out.append((it, s))

#     return out

def sample_all_windows_npz(
    ds: "NpzVideoDataset",
    clip_len: int = 32,
    stride:   int = 8,
):
    """
    Return all valid (VideoItem, start) windows from the dataset.
    - If video has >= clip_len frames: slide with stride.
    - If video has <  clip_len frames: include one window covering all frames.
    """
    out: T.List[T.Tuple["VideoItem", int]] = []

    for it in tqdm(ds.items, desc="Enumerating all windows", total=len(ds.items)):
        if it.length < clip_len:
            # Short video → one full-length window
            out.append((it, 0))
            continue

        max_start = it.length - clip_len
        starts = range(0, max_start + 1, stride)
        for s in starts:
            out.append((it, s))

    return out

def _enumerate_windows_for_item(it: VideoItem, clip_len: int = 32, stride: int = 8):
    out = []
    max_start = it.length - clip_len
    if max_start < 0:
        return out
    for s in range(0, max_start + 1, stride):
        out.append((it, s))
    return out

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

        assert len(self.classes) >= self.P, f"P: {self.P} exceeds num classes: {len(self.classes)}"

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


@torch.no_grad()
def build_train_centroids_subset(model, small_loader, label_dict, device, feature_selector=None):
    model.eval()
    C = len(label_dict)
    sums, counts = None, torch.zeros(C, device=device, dtype=torch.float32)

    with torch.no_grad():
        for packed in tqdm(small_loader, desc="Building centroids", total=len(small_loader)):
            if packed is None:
                continue
            feats, cls_names, _ = packed
            feats = feats.to(device, non_blocking=True)
            if feature_selector is not None:
                feats = feature_selector.select(feats)
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

def get_human_corr(
    human_corr_meshes,
    human_scores_path: str,
    centroids_sub,
    label_dict: dict,
    stats,
    model,
    clip_len: int,
    stride: int,
    gen_kp_dir: str = None,
    gen_clip_dir: str = None,
    gen_dino_dir: str = None,
):
    """
    Load human scores from single JSON file and compute correlations.
    JSON format: {video_name.mp4: {"ac": score, "tc": score}}
    Returns: (out_appearance, out_action, out_anatomy, out_motion) dicts
    """
    with open(human_scores_path, 'r') as f:
        human_scores = json.load(f)

    # --------- Build VideoItems from provided mesh paths ----------
    def _infer_class_from_name(name: str, known_classes: T.Iterable[str]) -> str:
        for cls in known_classes:
            if cls in name:
                return cls
        return next(iter(known_classes)) if known_classes else "unknown"

    items: T.List[VideoItem] = []
    for mesh_path in human_corr_meshes:
        base = os.path.basename(mesh_path)
        name_noext = os.path.splitext(base)[0]
        try:
            npz = np.load(mesh_path, mmap_mode="r")
            vit = npz["vit"]
            length = vit.shape[0]
            vit_dim = vit.shape[1]
        except Exception:
            continue

        cls_name = _infer_class_from_name(name_noext, label_dict.keys())
        items.append(VideoItem(cls=cls_name, name=base, path=mesh_path, length=length, vit_dim=vit_dim))

    if not items:
        out_none = {"spearman": None, "pearson": None, "n": 0}
        return out_none, out_none, out_none, out_none

    dataset = NpzVideoDataset(root_dir=os.path.commonpath([it.path for it in items]) or ".", items=items, enforce_min_per_class=False)
    samples = sample_all_windows_npz(dataset, clip_len=clip_len, stride=stride)
    window_ds = WindowDataset(samples, clip_len=clip_len, stats=stats, keypoint_dir=gen_kp_dir, clip_dir=gen_clip_dir, dino_dir=gen_dino_dir, seed=SEED)

    loader = DataLoader(
        window_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=safe_collate,
        pin_memory=False,
    )

    device = next(model.parameters()).device
    model.eval()

    def _norm_name(name: str) -> str:
        stem = os.path.splitext(os.path.basename(name))[0]
        stem = stem.replace("_videos_", "_")
        stem = stem.replace("videos_", "")
        stem = stem.replace("_video_", "_")
        return stem
    def _extract_class(name: str) -> T.Optional[str]:
        for cls in label_dict.keys():
            if cls in name:
                return cls
        return None

    # --------- Compute both motion and action scores in a single pass ---------
    video_to_scores: T.Dict[str, T.List[float]] = {}
    video_to_cls_embeds: T.Dict[str, T.List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            feats, cls_names, vid_names = batch
            feats = feats.to(device)
            emb, frame_emb, _ = model(feats)
            if frame_emb is None:
                continue

            # Motion (temporal coherence): frame_emb: [B, T+1, dim] where [:, 0, :] is CLS token, [:, 1:, :] are frame tokens
            # Compute mean L2 consecutive frame diff per window (excluding CLS token)
            frame_tokens = frame_emb[:, 1:, :]  # [B, T, dim] - exclude CLS
            diffs = (frame_tokens[:, 1:] - frame_tokens[:, :-1]).pow(2).sum(dim=-1).sqrt().mean(dim=-1)
            for vid_name, score in zip(vid_names, diffs.detach().cpu().numpy().tolist()):
                video_to_scores.setdefault(vid_name, []).append(score)

            # Action consistency: collect CLS embeddings
            for cls_name, vid_name, cls_emb in zip(cls_names, vid_names, emb):
                video_to_cls_embeds.setdefault(vid_name, []).append(cls_emb.detach().cpu())

    # --------- Motion (temporal coherence) per video ---------
    motion_scores = {k: float(np.mean(v)) for k, v in video_to_scores.items() if len(v) > 0}

    # --------- Action consistency (distance to class centroid) ---------

    action_scores: T.Dict[str, float] = {}
    for vid_name, embeds in video_to_cls_embeds.items():
        cls_name = _extract_class(_norm_name(vid_name))
        if cls_name is None or cls_name not in label_dict:
            continue
        idx = label_dict[cls_name]
        if idx >= len(centroids_sub):
            continue
        centroid = centroids_sub[idx].detach().cpu()
        z_mean = torch.stack(embeds, dim=0).mean(dim=0)
        z_mean = F.normalize(z_mean, p=2, dim=-1)  # Normalize before computing distance
        action_scores[vid_name] = float(torch.norm(z_mean - centroid, p=2).item())

    def compute_correlations(model_scores_dict: dict, human_key: str):
        """Compute correlations between model scores and human scores."""
        model_by_name = {_norm_name(k): v for k, v in model_scores_dict.items()}

        class_scores: T.Dict[str, T.List[float]] = {}
        for k, v in model_by_name.items():
            cls = _extract_class(k)
            if cls is not None:
                class_scores.setdefault(cls, []).append(v)
        class_means = {c: float(np.mean(vs)) for c, vs in class_scores.items() if vs}

        model_values = []
        human_values = []

        for human_key_name, human_data in human_scores.items():
            if human_key not in human_data:
                continue

            human_name_norm = _norm_name(human_key_name)
            human_cls = _extract_class(human_name_norm)

            if human_name_norm in model_by_name:
                model_values.append(model_by_name[human_name_norm])
                human_values.append(human_data[human_key])
                continue

            if human_cls and human_cls in class_means:
                model_values.append(class_means[human_cls])
                human_values.append(human_data[human_key])

        if len(model_values) < 2:
            return {"spearman": None, "pearson": None, "n": len(model_values)}

        model_array = np.array(model_values)
        human_array = np.array(human_values)

        spearman_corr, _ = spearmanr(model_array, human_array)
        pearson_corr, _ = pearsonr(model_array, human_array)

        # Invert sign: model scores are "lower is better" (distances), human scores are "higher is better"
        # So we need negative correlation to indicate good alignment
        spearman_corr = -spearman_corr if not np.isnan(spearman_corr) else spearman_corr
        pearson_corr = -pearson_corr if not np.isnan(pearson_corr) else pearson_corr

        return {
            "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else None,
            "pearson": float(pearson_corr) if not np.isnan(pearson_corr) else None,
        }

    out_action = compute_correlations(action_scores, "ac")
    out_motion = compute_correlations(motion_scores, "tc")
    out_appearance = {"spearman": None, "pearson": None, "n": 0}
    out_anatomy = {"spearman": None, "pearson": None, "n": 0}

    return out_appearance, out_action, out_anatomy, out_motion