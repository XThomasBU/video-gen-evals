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
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np

import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import json

def build_video_split(root, classes, train_ratio=0.8, seed=1, group_by_group=True):
    """
    Build a single, reproducible split and return:
        split_map = { "train": {cls: [Path, ...]}, "test": {cls: [Path, ...]} }
    If group_by_group=True, keep all clips from the same 'gXX' together.
    """
    rng = np.random.default_rng(seed)
    split_map = {"train": {c: []}, "test": {c: []}}
    for c in classes:
        class_dir = Path(root) / c
        vids = sorted([class_dir / v for v in os.listdir(class_dir)])

        if group_by_group:
            # Group by gXX (e.g., v_Class_g19_c03 -> group 'g19')
            groups = defaultdict(list)
            for p in vids:
                m = re.search(r"_g(\d+)_", p.name)
                gid = m.group(1) if m else "ungrouped"
                groups[gid].append(p)

            group_ids = list(groups.keys())
            rng.shuffle(group_ids)

            n_train_groups = int(train_ratio * len(group_ids))
            train_gids = set(group_ids[:n_train_groups])

            for gid, items in groups.items():
                (split_map["train" if gid in train_gids else "test"].setdefault(c, [])).extend(items)
        else:
            vids = list(vids)
            rng.shuffle(vids)
            n_train = int(train_ratio * len(vids))
            split_map["train"].setdefault(c, []).extend(vids[:n_train])
            split_map["test"].setdefault(c, []).extend(vids[n_train:])

    return split_map

# set seed
torch.manual_seed(1)
np.random.seed(1)

# ——— CONFIG ———
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 100
WINDOW_SIZE = 32 # 64
STRIDE = 8 # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES"

INPUT_DIM= 1370

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

import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
from tqdm import tqdm

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


# ======================================
# Stats helpers (compute mean/std safely)
# ======================================

def _accumulate(arr_list, x: torch.Tensor):
    if len(arr_list) == 0:
        arr_list.append(x.detach().cpu())
    else:
        arr_list.append(x.detach().cpu())


def _finalize_stats(np_list: list):
    if len(np_list) == 0:
        return None, None
    X = torch.cat(np_list, dim=0).numpy()
    return X.mean(axis=0), X.std(axis=0)


# ============================================================
# Global stats (raw + motion) — compute from TRAINING videos
# ============================================================

def get_global_stats(classes, full_videos, pose_dir, save_dir: str):
    """
    Computes and saves mean/std for both RAW and MOTION features, per modality.
    IMPORTANT: Call this ONLY on training data to avoid leakage.

    Saves files to save_dir with keys:
        RAW:    vit_mean/std, global_orient_mean/std, body_pose_mean/std, betas_mean/std, twod_kp_mean/std
        MOTION: vit_d_mean/std, global_orient_d_mean/std, body_pose_d_mean/std, betas_d_mean/std, twod_kp_d_mean/std
    """
    save = Path(save_dir)
    save.mkdir(parents=True, exist_ok=True)

    raw_buffers = {k: [] for k in ['vit', 'global_orient', 'body_pose', 'betas', 'twod_kp']}
    d_buffers   = {k: [] for k in ['vit_d', 'global_orient_d', 'body_pose_d', 'betas_d', 'twod_kp_d']}

    for cls in classes:
        videos = full_videos[cls]
        for video_folder in tqdm(videos, desc=f"Stats ({cls})", total=len(videos)):
            frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
            twod_points_dir = str(video_folder).replace(
                "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh",
                pose_dir,
            )
            twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))
            T = min(len(frames), len(twod_points_paths))
            if T < 2:
                continue

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

                vit = np.asarray(params["token_out"]).reshape(-1)           # e.g., 1024
                go  = np.asarray(params["global_orient"]).reshape(-1)       # 9 (R as flat)
                pose = np.asarray(params["body_pose"]).reshape(-1)          # 207 (axis-angle)
                bet  = np.asarray(params["betas"]).reshape(-1)              # 10
                kp   = np.load(twod_points_paths[t]).reshape(-1)[:120]       # 120

                vit_list.append(torch.tensor(vit, dtype=torch.float32))
                go_list.append(torch.tensor(go, dtype=torch.float32))
                pose_list.append(torch.tensor(pose, dtype=torch.float32))
                betas_list.append(torch.tensor(bet, dtype=torch.float32))
                kp_list.append(torch.tensor(kp, dtype=torch.float32))

            if len(vit_list) == 0:
                continue

            vit   = torch.stack(vit_list, dim=0)
            go    = torch.stack(go_list, dim=0)
            pose  = torch.stack(pose_list, dim=0)
            betas = torch.stack(betas_list, dim=0)
            kp2d  = torch.stack(kp_list, dim=0)

            # --- accumulate RAW ---
            _accumulate(raw_buffers['vit'], vit)
            _accumulate(raw_buffers['global_orient'], go)
            _accumulate(raw_buffers['body_pose'], pose)
            _accumulate(raw_buffers['betas'], betas)
            _accumulate(raw_buffers['twod_kp'], kp2d)

            # --- compute MOTION on raw geometry, then accumulate ---
            d_vit  = _vit_delta(vit)
            d_go   = _rot_matrix_delta(go)
            d_pose = _rot_axisangle_delta(pose)
            d_bet  = _betas_delta(betas)
            d_kp   = _procrustes_kp_delta(kp2d)

            _accumulate(d_buffers['vit_d'], d_vit)
            _accumulate(d_buffers['global_orient_d'], d_go)
            _accumulate(d_buffers['body_pose_d'], d_pose)
            _accumulate(d_buffers['betas_d'], d_bet)
            _accumulate(d_buffers['twod_kp_d'], d_kp)

    # finalize and save
    stats = {}
    # raw
    for key in ['vit', 'global_orient', 'body_pose', 'betas', 'twod_kp']:
        m, s = _finalize_stats(raw_buffers[key])
        stats[f'{key}_mean'] = m
        stats[f'{key}_std'] = s
        if m is not None:
            np.save(save / f'{key}_mean.npy', m)
            np.save(save / f'{key}_std.npy', s)
    # motion
    for key in ['vit_d', 'global_orient_d', 'body_pose_d', 'betas_d', 'twod_kp_d']:
        m, s = _finalize_stats(d_buffers[key])
        stats[f'{key}_mean'] = m
        stats[f'{key}_std'] = s
        if m is not None:
            np.save(save / f'{key}_mean.npy', m)
            np.save(save / f'{key}_std.npy', s)

    return stats


# ======================================================
# Sequence loader (delta first, then normalize)
# ======================================================

def load_video_sequence(video_folder, stats_dir, pose_dir, expect_dim=1370):
    """
    Builds [T, 2740] = [T, 1370 raw | 1370 motion] with *geometry-respecting* deltas.

    Normalization order:
      - ROTATIONS/KEYPOINTS/BETAS: compute deltas on raw geometry -> z-score with MOTION stats
      - ViT: L2-normalize per frame before diff (cosine change), then z-score with MOTION stats
      - RAW branches are z-scored with RAW stats

    Expected per-modality dims (adapt if different in your setup):
      vit: 1024, global_orient: 9 (R flat), body_pose: 207 (axis-angle), betas: 10, twod_kp: 120
    """
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

    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    twod_points_dir = str(video_folder).replace(
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh",
        pose_dir,
    )
    twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

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

        vit  = np.asarray(params["token_out"]).reshape(-1)
        go   = np.asarray(params["global_orient"]).reshape(-1)
        pose = np.asarray(params["body_pose"]).reshape(-1)
        bet  = np.asarray(params["betas"]).reshape(-1)
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

    # Sanity check on expected dims (optional)
    if expect_dim is not None:
        assert raw.shape[-1] == expect_dim, f"Expected raw dim {expect_dim}, got {raw.shape[-1]}"

    return enriched  # [T, 2*expect_dim]


# ——— SLIDING WINDOW ———
def extract_windows(seq, window_size, stride):
    windows = []
    num_frames = seq.shape[0]
    for start in range(0, num_frames, stride):
        end = start + window_size
        if end > num_frames:
            # pad_len = end - num_frames
            # pad = seq[-1:].repeat(pad_len, 1)
            # window = torch.cat([seq[start:], pad], dim=0)
            continue
        else:
            window = seq[start:end]
        windows.append(window)
        if end >= num_frames:
            break
    return windows

class PoseVideoDataset(Dataset):
    def __init__(self, root, classes, window_size=64, stride=32, split="train"):
        self.samples = []
        self.labels = []
        self.vid_ids = []
        self.window_ids = []  # Store window IDs for each sample
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.per_class_video_paths = {c: [] for c in classes}
        for cls in classes:
            class_dir = Path(root) / cls
            videos = [class_dir / vid for vid in os.listdir(class_dir)]
            self.per_class_video_paths[cls] = videos

        # Split videos *per class*
        self.video_split = {}
        for cls in classes:
            vids = self.per_class_video_paths[cls]
            rng = np.random.RandomState(1)  # fixed seed for determinism
            rng.shuffle(vids)               # same shuffle for both train/test
            n_train = int(0.8 * len(vids))
            if split == "train":
                selected_videos = vids[:n_train]
            else:
                selected_videos = vids[n_train:]
            self.video_split[cls] = selected_videos

        # print(len(self.video_split), "classes with video splits for", split)
        # # if split == "train":
        # # if os.path.exists("SAVE/betas_mean.py"):
        # # global_stats = {}
        # # global_stats["vit_mean"] = np.load("SAVE/vit_mean.npy")
        # # global_stats["vit_std"] = np.load("SAVE/vit_std.npy")
        # # global_stats["global_orient_mean"] = np.load("SAVE/global_orient_mean.npy")
        # # global_stats["global_orient_std"] = np.load("SAVE/global_orient_std.npy")
        # # global_stats["betas_mean"] = np.load("SAVE/betas_mean.npy")
        # # global_stats["betas_std"] = np.load("SAVE/betas_std.npy")
        # # global_stats["twod_kp_mean"] = np.load("SAVE/twod_kp_mean.npy")
        # # global_stats["twod_kp_std"] = np.load("SAVE/twod_kp_std.npy")
        # # else:
        # global_stats = get_global_stats(classes, self.video_split, POSE_DIR, "SAVE")
        # exit()

        # Now extract all windows from the selected videos
        for cls in tqdm(classes, desc=f"Loading {split} videos", total=len(classes)):
            videos = self.video_split[cls]
            for idx, vid_path in enumerate(tqdm(videos, desc=f"Loading {cls} videos", total=len(videos))):
                seq = load_video_sequence(vid_path, "SAVE", POSE_DIR)
                if seq is None:
                    continue
                windows = extract_windows(seq, WINDOW_SIZE, STRIDE)
                self.samples.extend(windows)
                self.labels.extend([self.class_to_idx[cls]] * len(windows))
                self.vid_ids.extend([vid_path.name] * len(windows))
                self.window_ids.extend(list(range(len(windows))))
                # if idx == 9:
                #     break

        print(f" Loaded {len(self.samples)} {split} windows from {split} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.vid_ids[idx], self.window_ids[idx]


def main():

    train_dataset = PoseVideoDataset(
        REAL_ROOT,
        ALL_CLASSES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split="train"
    )
    test_dataset = PoseVideoDataset(
        REAL_ROOT,
        ALL_CLASSES,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split="test"
    )

    # save dataset as tensors
    torch.save(train_dataset.samples, f"SAVE/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(train_dataset.labels, f"SAVE/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.samples, f"SAVE/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.labels, f"SAVE/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(train_dataset.vid_ids, f"SAVE/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.vid_ids, f"SAVE/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(train_dataset.window_ids, f"SAVE/train_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    torch.save(test_dataset.window_ids, f"SAVE/test_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    print(f" Created new datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")


# ——— MAIN ———
if __name__ == "__main__":
    main()