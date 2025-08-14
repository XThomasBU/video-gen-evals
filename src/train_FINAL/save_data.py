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
WINDOW_SIZE = 64 # 64
STRIDE = 16 # 32
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

def get_global_stats(classes, full_videos):
    global_stats = {}
    for cls in classes:
        videos = full_videos[cls]
        for idx, video_folder in enumerate(tqdm(videos, desc=f"Loading {cls} videos to get global mean", total=len(videos))):

            frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))

            twod_points_dir = str(video_folder).replace("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh", POSE_DIR)
            twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

            collect_inputs = {
                'vit': [],
                'global_orient': [],
                'body_pose': [],
                'betas': [],
                'twod_kp': []
            }

            for idx, p in enumerate(frames):
                # try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                params = data["pred_smpl_params"]

                if isinstance(params, list):
                    if len(params) < 1:
                        continue
                    params = params[0]
                if not isinstance(params, dict):
                    continue

                vit_feature   = np.array(params["token_out"]).flatten()
                global_orient = np.array(params["global_orient"]).flatten()
                body_pose     = np.array(params["body_pose"]).flatten()
                betas         = np.array(params["betas"]).flatten()

                twod_point_path = twod_points_paths[idx]
                twod_kp = np.load(twod_point_path).flatten()
                twod_kp = twod_kp[:120]  # Ensure 120 keypoints

                collect_inputs['vit'].append(vit_feature)
                collect_inputs['global_orient'].append(global_orient)
                collect_inputs['body_pose'].append(body_pose)
                collect_inputs['betas'].append(betas)
                collect_inputs['twod_kp'].append(twod_kp)

    # get mean and std dev of each 
    global_stats['vit_mean'] = np.mean(collect_inputs['vit'], axis=0)
    global_stats['vit_std'] = np.std(collect_inputs['vit'], axis=0)

    global_stats['global_orient_mean'] = np.mean(collect_inputs['global_orient'], axis=0)
    global_stats['global_orient_std'] = np.std(collect_inputs['global_orient'], axis=0)

    global_stats['body_pose_mean'] = np.mean(collect_inputs['body_pose'], axis=0)
    global_stats['body_pose_std'] = np.std(collect_inputs['body_pose'], axis=0)

    global_stats['betas_mean'] = np.mean(collect_inputs['betas'], axis=0)
    global_stats['betas_std'] = np.std(collect_inputs['betas'], axis=0)

    global_stats['twod_kp_mean'] = np.mean(collect_inputs['twod_kp'], axis=0)
    global_stats['twod_kp_std'] = np.std(collect_inputs['twod_kp'], axis=0)

    # save global stats as numpy files
    for key, value in global_stats.items():
        np.save(f"SAVE/{key}.npy", value)

    return global_stats

# ——— LOAD VIDEO FRAMES ———
def load_video_sequence(video_folder, global_stats_file):

    global_stats = {}
    for key in ['vit_mean', 'vit_std', 'global_orient_mean', 'global_orient_std',
                'body_pose_mean', 'body_pose_std', 'betas_mean', 'betas_std',
                'twod_kp_mean', 'twod_kp_std']:
        global_stats[key] = np.load(f"{global_stats_file}/{key}.npy")

    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh", POSE_DIR)
    twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

    for idx, p in enumerate(frames):
        # try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        params = data["pred_smpl_params"]

        if isinstance(params, list):
            if len(params) < 1:
                continue
            params = params[0]
        if not isinstance(params, dict):
            continue

        vit_feature   = np.array(params["token_out"]).flatten()
        global_orient = np.array(params["global_orient"]).flatten()
        body_pose     = np.array(params["body_pose"]).flatten()
        betas         = np.array(params["betas"]).flatten()

        twod_point_path = twod_points_paths[idx]
        twod_kp = np.load(twod_point_path).flatten()
        twod_kp = twod_kp[:120]  # Ensure 120 keypoints

        # # Normalize each part
        vit_feature = (vit_feature - global_stats['vit_mean']) / (global_stats['vit_std'] + 1e-8)
        global_orient = (global_orient - global_stats['global_orient_mean']) / (global_stats['global_orient_std'] + 1e-8)
        body_pose     = (body_pose - global_stats['body_pose_mean']) / (global_stats['body_pose_std'] + 1e-8)
        betas         = (betas - global_stats['betas_mean']) / (global_stats['betas_std'] + 1e-8)
        twod_kp       = (twod_kp - global_stats['twod_kp_mean']) / (global_stats['twod_kp_std'] + 1e-8)

        vec = np.concatenate([vit_feature, global_orient, body_pose, betas, twod_kp], axis=0)

        frame_vecs.append(torch.tensor(vec, dtype=torch.float32))

    if len(frame_vecs) < 2:
        return None

    # [T, 1250]
    frame_tensor = torch.stack(frame_vecs, dim=0)

    # Compute motion vectors (frame-to-frame deltas)
    motion_vecs = frame_tensor[1:] - frame_tensor[:-1]  # [T-1, 1250]
    motion_vecs = torch.cat([torch.zeros(1, INPUT_DIM), motion_vecs], dim=0)  # [T, 1250]


    # Concatenate original + motion
    enriched_tensor = torch.cat([frame_tensor, motion_vecs], dim=1)  # [T, 2500] # 2500 = 1250 * 2
    return enriched_tensor

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

        print(len(self.video_split), "classes with video splits for", split)
        if split == "train":
            global_stats = get_global_stats(classes, self.video_split)
            

        # Now extract all windows from the selected videos
        for cls in tqdm(classes, desc=f"Loading {split} videos", total=len(classes)):
            videos = self.video_split[cls]
            for idx, vid_path in enumerate(tqdm(videos, desc=f"Loading {cls} videos", total=len(videos))):
                seq = load_video_sequence(vid_path, "SAVE")
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