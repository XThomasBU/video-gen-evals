import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from dataclasses import fields
from utils import (
    ModalityStats, NpzVideoDataset, VideoItem, WindowDataset,
    sample_all_windows_npz, safe_collate, seed_worker, DEVICE,
    compute_stats_from_npz, build_train_centroids_subset, make_test_loader,
    train_test_split,
)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model import HumanActionScorer
import json
from collections import defaultdict
from scipy.stats import spearmanr


ACTION_CLASSES = [
    "BodyWeightSquats",
    "HulaHoop",
    "JumpingJack",
    "PullUps",
    "PushUps",
    "Shotput",
    "SoccerJuggling",
    "TennisSwing",
    "ThrowDiscus",
    "WallPushups",
]


def _canonicalize_class(name: str) -> str:
    """Map filename class tokens onto the canonical ACTION_CLASSES labels."""
    for cls in ACTION_CLASSES:
        if name.lower() == cls.lower():
            return cls
    aliases = {
        "soccerjuggling": "SoccerJuggling",
        "tennisswing": "TennisSwing",
    }
    return aliases.get(name.lower(), name)


def create_dataset_from_generated_meshes(generated_meshes_dir: str) -> NpzVideoDataset:
    items = []
    generated_meshes_path = Path(generated_meshes_dir)

    for npz_file in tqdm(sorted(generated_meshes_path.glob("*.npz")), desc="Scanning generated_meshes"):
        try:
            npz = np.load(npz_file, mmap_mode="r")
            stem = npz_file.stem
            parts = stem.split("_")

            cls_name = None
            for part in parts:
                canon = _canonicalize_class(part)
                if canon in ACTION_CLASSES:
                    cls_name = canon
                    break

            if cls_name is None:
                for part in parts:
                    if (part[0].isupper() and not part.isdigit() and
                        len(part) > 3 and part.lower() not in ["videos", "npz"]):
                        cls_name = _canonicalize_class(part)
                        break

            if cls_name is None:
                cls_name = "Unknown"

            if "pose" in npz:
                Tlen = npz["pose"].shape[0]
            else:
                Tlen = 0

            if "vit" in npz:
                vit = npz["vit"]
                Dvit = vit.shape[1] if len(vit.shape) > 1 else 0
            else:
                Dvit = 0

            items.append(VideoItem(
                cls=cls_name,
                name=npz_file.name,
                path=str(npz_file),
                length=Tlen,
                vit_dim=Dvit,
            ))
        except Exception as e:
            print(f"Failed to load {npz_file}: {e}")
            continue

    return NpzVideoDataset(
        root_dir=generated_meshes_dir,
        items=items,
        enforce_min_per_class=False,
    )


def infer_dims_from_stats(stats: ModalityStats) -> tuple:
    # Key names match train.py / checkpoint ModuleDict keys.
    dims_map_raw = {
        "vit": stats.vit_raw_mean.shape[0] if stats.vit_raw_mean is not None else 0,
        "global": stats.gori_raw_mean.shape[0] if stats.gori_raw_mean is not None else 0,
        "pose": stats.pose_raw_mean.shape[0] if stats.pose_raw_mean is not None else 0,
        "beta": stats.beta_raw_mean.shape[0] if stats.beta_raw_mean is not None else 0,
    }
    dims_map_diff = {
        "vit": stats.vit_diff_mean.shape[0] if stats.vit_diff_mean is not None else 0,
        "global": stats.gori_diff_mean.shape[0] if stats.gori_diff_mean is not None else 0,
        "pose": stats.pose_diff_mean.shape[0] if stats.pose_diff_mean is not None else 0,
        "beta": stats.beta_diff_mean.shape[0] if stats.beta_diff_mean is not None else 0,
    }

    if stats.keypoints_raw_mean is not None:
        dims_map_raw["kp2d"] = stats.keypoints_raw_mean.shape[0]
        dims_map_diff["kp2d"] = (
            stats.keypoints_diff_mean.shape[0] if stats.keypoints_diff_mean is not None else 0
        )

    if stats.clip_raw_mean is not None:
        dims_map_raw["clip"] = stats.clip_raw_mean.shape[0]
        dims_map_diff["clip"] = stats.clip_diff_mean.shape[0] if stats.clip_diff_mean is not None else 0

    if stats.dino_raw_mean is not None:
        dims_map_raw["dino"] = stats.dino_raw_mean.shape[0]
        dims_map_diff["dino"] = stats.dino_diff_mean.shape[0] if stats.dino_diff_mean is not None else 0

    return dims_map_raw, dims_map_diff


def load_model(model_path: str, dims_map_raw: dict, dims_map_diff: dict, device=DEVICE):
    checkpoint = torch.load(model_path, map_location=device)

    d_model = checkpoint.get("d_model", 256) if isinstance(checkpoint, dict) else 256
    latent_dim = checkpoint.get("latent_dim", 128) if isinstance(checkpoint, dict) else 128
    time_layers = checkpoint.get("time_layers", 4) if isinstance(checkpoint, dict) else 4
    time_heads = checkpoint.get("time_heads", 8) if isinstance(checkpoint, dict) else 8
    dropout = checkpoint.get("dropout", 0.1) if isinstance(checkpoint, dict) else 0.1

    model = HumanActionScorer(
        dims_map_raw=dims_map_raw,
        dims_map_diff=dims_map_diff,
        d_model=d_model,
        latent_dim=latent_dim,
        time_layers=time_layers,
        time_heads=time_heads,
        dropout=dropout,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_window_features(
    model: nn.Module,
    dataloader: DataLoader,
    device=DEVICE,
    save_path: str = None,
):
    all_seq_embeds = []
    all_frame_embeds = []
    all_cls_names = []
    all_vid_names = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if batch is None:
                continue

            feats, cls_names, vid_names = batch
            feats = feats.to(device, non_blocking=True)

            seq_embed, frame_embeds, _tokens = model(feats)

            all_seq_embeds.append(seq_embed.cpu())
            all_frame_embeds.append(frame_embeds.cpu())
            all_cls_names.extend(cls_names)
            all_vid_names.extend(vid_names)

    features = {
        "seq_embeds": torch.cat(all_seq_embeds, dim=0),
        "frame_embeds": torch.cat(all_frame_embeds, dim=0),
        "cls_names": all_cls_names,
        "vid_names": all_vid_names,
    }

    if save_path:
        torch.save(features, save_path)
        print(f"Saved features to {save_path}")

    return features


def compute_temporal_coherence_scores(features: dict):
    """TC: mean consecutive L2 distance between per-frame embeddings (exclude CLS)."""
    frame_embeds = features["frame_embeds"]
    vid_names = features["vid_names"]

    video_scores = defaultdict(list)

    for i, vid_name in enumerate(vid_names):
        video_id = os.path.splitext(vid_name)[0]
        frames_only = frame_embeds[i][1:]

        if frames_only.shape[0] < 2:
            continue

        diffs = (frames_only[1:] - frames_only[:-1]).pow(2).sum(dim=-1).sqrt()
        video_scores[video_id].append(float(diffs.mean().item()))

    return {vid: float(np.mean(scores)) for vid, scores in video_scores.items()}


def compute_action_consistency_scores(features: dict, centroids, label_dict: dict):
    """AC: L2 distance from mean window embedding to the real-action class centroid."""
    seq_embeds = features["seq_embeds"]
    cls_names = features["cls_names"]
    vid_names = features["vid_names"]

    video_to_embeds = defaultdict(list)
    video_to_cls = {}

    for i, vid_name in enumerate(vid_names):
        video_id = os.path.splitext(vid_name)[0]
        cls_name = _canonicalize_class(cls_names[i])
        video_to_embeds[video_id].append(seq_embeds[i])
        video_to_cls[video_id] = cls_name

    action_scores = {}
    for video_id, embeds in video_to_embeds.items():
        cls_name = video_to_cls[video_id]
        if cls_name not in label_dict:
            continue
        idx = label_dict[cls_name]
        if idx >= len(centroids):
            continue

        z_mean = F.normalize(torch.stack(embeds, dim=0).mean(dim=0), p=2, dim=-1)
        centroid = centroids[idx].detach().cpu()
        action_scores[video_id] = float(torch.norm(z_mean - centroid, p=2).item())

    return action_scores


def build_real_centroids(
    model,
    real_meshes_dir: str,
    real_kp_dir: str,
    stats: ModalityStats,
    clip_len: int = 32,
    stride: int = 8,
    device=DEVICE,
):
    real_ds = NpzVideoDataset(real_meshes_dir, filter_classes=ACTION_CLASSES)
    train_ds, _ = train_test_split(real_ds, train_ratio=0.8, seed=1337)
    label_dict = {cls: i for i, cls in enumerate(sorted({it.cls for it in real_ds.items}))}

    print(f"Building centroids from {len(train_ds)} real videos across {len(label_dict)} classes...")
    real_loader = make_test_loader(
        train_ds,
        clip_len=clip_len,
        stride=stride,
        stats=stats,
        seed=1337,
        batch_size=64,
        keypoint_dir=real_kp_dir,
        num_workers=0,
    )
    centroids, counts = build_train_centroids_subset(model, real_loader, label_dict, device=device)
    print(f"Centroid sample counts: {counts.detach().cpu().tolist()}")
    return centroids, label_dict


def _norm_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = stem.replace("_videos_", "_")
    stem = stem.replace("videos_", "")
    stem = stem.replace("_video_", "_")
    return stem


def compute_spearman_correlation(model_scores: dict, human_scores_path: str, human_key: str):
    """
    Spearman correlation between model scores and human ratings.
    Model scores are distances (lower better); human scores are higher-better,
    so the reported correlation is sign-inverted to match training eval.
    """
    with open(human_scores_path, "r") as f:
        human_scores = json.load(f)

    model_by_name = {_norm_name(k): v for k, v in model_scores.items()}

    model_values = []
    human_values = []
    matched_videos = []

    for human_key_name, human_data in human_scores.items():
        if human_key not in human_data:
            continue

        human_name_norm = _norm_name(human_key_name)

        if human_name_norm in model_by_name:
            model_values.append(model_by_name[human_name_norm])
            human_values.append(human_data[human_key])
            matched_videos.append((human_name_norm, human_key_name))
            continue

        human_parts = human_name_norm.split("_")
        for model_name_norm, model_score in model_by_name.items():
            model_parts = model_name_norm.split("_")
            if len(model_parts) >= 2 and len(human_parts) >= 2:
                if model_parts[-2:] == human_parts[-2:] or model_parts[-1] == human_parts[-1]:
                    model_values.append(model_score)
                    human_values.append(human_data[human_key])
                    matched_videos.append((model_name_norm, human_key_name))
                    break

    if len(model_values) < 2:
        print(f"Warning: Only {len(model_values)} matched videos for {human_key}. Need at least 2.")
        return None, None, matched_videos

    correlation, p_value = spearmanr(np.array(model_values), np.array(human_values))
    if correlation is not None and not np.isnan(correlation):
        correlation = -float(correlation)

    print(f"\nSpearman Correlation ({human_key.upper()}):")
    print(f"Matched videos: {len(matched_videos)}")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")

    return correlation, p_value, matched_videos


if __name__ == "__main__":
    generated_meshes_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/generated_meshes"
    real_meshes_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/meshes_10classes"
    model_path = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/model.pt"
    keypoint_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/generated_kps"
    real_kp_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/SAVE_REAL_ONLY_10_minus1"
    human_scores_path = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/human_scores.json"

    clip_len = 32
    stride = 8

    print("Loading real meshes for stats / AC centroids...")
    if not os.path.isdir(real_meshes_dir):
        raise FileNotFoundError(
            f"Real meshes required for AC scoring not found: {real_meshes_dir}"
        )

    real_ds = NpzVideoDataset(real_meshes_dir, filter_classes=ACTION_CLASSES)
    train_ds, _ = train_test_split(real_ds, train_ratio=0.8, seed=1337)
    print(f"Loaded {len(real_ds)} real videos ({len(train_ds)} train) across {sorted(real_ds.classes)}")

    print("Computing modality stats from real train set...")
    stats = compute_stats_from_npz(train_ds.items, keypoint_dir=real_kp_dir)
    print(f"Computed stats with {len([f for f in fields(ModalityStats)])} fields")

    dims_map_raw, dims_map_diff = infer_dims_from_stats(stats)
    print(f"dims_map_raw: {dims_map_raw}")
    print(f"dims_map_diff: {dims_map_diff}")

    print("Loading model...")
    model = load_model(model_path, dims_map_raw, dims_map_diff)
    print("Model loaded successfully")

    centroids, label_dict = build_real_centroids(
        model=model,
        real_meshes_dir=real_meshes_dir,
        real_kp_dir=real_kp_dir,
        stats=stats,
        clip_len=clip_len,
        stride=stride,
        device=DEVICE,
    )

    print("Loading dataset from generated_meshes...")
    dataset = create_dataset_from_generated_meshes(generated_meshes_dir)
    print(f"Loaded dataset with {len(dataset)} items")
    print(f"Classes: {sorted(dataset.classes)}")
    print(f"Items per class: {[(cls, len(dataset.class_to_items[cls])) for cls in sorted(dataset.classes)]}")

    print(f"Creating windows (clip_len={clip_len}, stride={stride})...")
    samples = sample_all_windows_npz(dataset, clip_len=clip_len, stride=stride)
    print(f"Created {len(samples)} windows")

    window_dataset = WindowDataset(
        samples=samples,
        clip_len=clip_len,
        stats=stats,
        keypoint_dir=keypoint_dir,
    )

    dataloader = DataLoader(
        window_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda"),
    )

    print("Extracting window features...")
    features = extract_window_features(
        model=model,
        dataloader=dataloader,
        save_path="window_features.pt",
    )

    print(f"Extracted features for {len(features['seq_embeds'])} windows")
    print(f"Sequence embeddings shape: {features['seq_embeds'].shape}")
    print(f"Frame embeddings shape: {features['frame_embeds'].shape}")

    print("Computing Action Consistency (AC) scores...")
    ac_scores = compute_action_consistency_scores(features, centroids, label_dict)
    print(f"AC scores for {len(ac_scores)} videos")

    print("Computing Temporal Coherence (TC) scores...")
    tc_scores = compute_temporal_coherence_scores(features)
    print(f"TC scores for {len(tc_scores)} videos")

    all_video_ids = sorted(set(ac_scores) | set(tc_scores))
    combined_scores = {}
    for vid in all_video_ids:
        entry = {}
        if vid in ac_scores:
            entry["ac"] = ac_scores[vid]
        if vid in tc_scores:
            entry["tc"] = tc_scores[vid]
        combined_scores[vid] = entry

    json_output_path = "video_scores.json"
    with open(json_output_path, "w") as f:
        json.dump(combined_scores, f, indent=2)

    print(f"Saved AC/TC scores for {len(combined_scores)} videos to {json_output_path}")
    print(f"Example scores (first 5): {dict(list(combined_scores.items())[:5])}")

    if os.path.exists(human_scores_path):
        print("\n" + "=" * 50)
        ac_corr, ac_p, _ = compute_spearman_correlation(ac_scores, human_scores_path, "ac")
        tc_corr, tc_p, _ = compute_spearman_correlation(tc_scores, human_scores_path, "tc")
        print("\nFinal Results:")
        if ac_corr is not None:
            print(f"AC Spearman: {ac_corr:.4f} (p={ac_p:.4e})")
        if tc_corr is not None:
            print(f"TC Spearman: {tc_corr:.4f} (p={tc_p:.4e})")
    else:
        print(f"\nWarning: Human scores file not found at {human_scores_path}")
