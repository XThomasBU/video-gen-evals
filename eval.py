import torch
import torch.nn as nn
import os
from pathlib import Path
from dataclasses import fields
from utils import (
    ModalityStats, NpzVideoDataset, VideoItem, WindowDataset,
    sample_all_windows_npz, safe_collate, seed_worker, DEVICE,
    compute_stats_from_npz
)
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model import HumanActionScorer
import json
from collections import defaultdict
from scipy.stats import spearmanr


def create_dataset_from_generated_meshes(generated_meshes_dir: str) -> NpzVideoDataset:
    items = []
    generated_meshes_path = Path(generated_meshes_dir)
    
    class_names = [
        "BodyWeightSquats", "HulaHoop", "JumpingJack", "PullUps", 
        "PushUps", "Shotput", "Soccerjuggling", "SoccerJuggling",
        "Tennisswing", "TennisSwing", "ThrowDiscus", "WallPushups"
    ]
    
    for npz_file in tqdm(sorted(generated_meshes_path.glob("*.npz")), desc="Scanning generated_meshes"):
        try:
            npz = np.load(npz_file, mmap_mode="r")
            stem = npz_file.stem
            parts = stem.split("_")
            
            cls_name = None
            for part in parts:
                if part in class_names:
                    cls_name = part
                    break
                for known_cls in class_names:
                    if part.lower() == known_cls.lower():
                        cls_name = known_cls
                        break
                if cls_name:
                    break
            
            if cls_name is None:
                for part in parts:
                    if (part[0].isupper() and not part.isdigit() and 
                        len(part) > 3 and part.lower() not in ['videos', 'npz']):
                        cls_name = part
                        break
            
            if cls_name is None:
                cls_name = "Unknown"
            
            if "pose" in npz:
                pose = npz["pose"]
                Tlen = pose.shape[0]
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
                vit_dim=Dvit
            ))
        except Exception as e:
            print(f"Failed to load {npz_file}: {e}")
            continue
    
    dataset = NpzVideoDataset(
        root_dir=generated_meshes_dir,
        items=items,
        enforce_min_per_class=False
    )
    
    return dataset


def infer_dims_from_stats(stats: ModalityStats) -> tuple:
    vit_dim = stats.vit_raw_mean.shape[0] if stats.vit_raw_mean is not None else 0
    gori_dim = stats.gori_raw_mean.shape[0] if stats.gori_raw_mean is not None else 0
    pose_dim = stats.pose_raw_mean.shape[0] if stats.pose_raw_mean is not None else 0
    beta_dim = stats.beta_raw_mean.shape[0] if stats.beta_raw_mean is not None else 0
    kp_dim = stats.keypoints_raw_mean.shape[0] if stats.keypoints_raw_mean is not None else 0
    
    dims_map_raw = {
        "vit": vit_dim,
        "gori": gori_dim,
        "pose": pose_dim,
        "beta": beta_dim,
        "keypoints": kp_dim
    }
    
    vit_diff_dim = stats.vit_diff_mean.shape[0] if stats.vit_diff_mean is not None else 0
    gori_diff_dim = stats.gori_diff_mean.shape[0] if stats.gori_diff_mean is not None else 0
    pose_diff_dim = stats.pose_diff_mean.shape[0] if stats.pose_diff_mean is not None else 0
    beta_diff_dim = stats.beta_diff_mean.shape[0] if stats.beta_diff_mean is not None else 0
    kp_diff_dim = stats.keypoints_diff_mean.shape[0] if stats.keypoints_diff_mean is not None else 0
    
    dims_map_diff = {
        "vit": vit_diff_dim,
        "gori": gori_diff_dim,
        "pose": pose_diff_dim,
        "beta": beta_diff_dim,
        "keypoints": kp_diff_dim
    }
    
    if stats.clip_raw_mean is not None:
        dims_map_raw["clip"] = stats.clip_raw_mean.shape[0]
        dims_map_diff["clip"] = stats.clip_diff_mean.shape[0] if stats.clip_diff_mean is not None else 0
    
    if stats.dino_raw_mean is not None:
        dims_map_raw["dino"] = stats.dino_raw_mean.shape[0]
        dims_map_diff["dino"] = stats.dino_diff_mean.shape[0] if stats.dino_diff_mean is not None else 0
    
    return dims_map_raw, dims_map_diff


def load_model(model_path: str, dims_map_raw: dict, dims_map_diff: dict, device=DEVICE):
    checkpoint = torch.load(model_path, map_location=device)
    
    d_model = checkpoint.get('d_model', 256) if isinstance(checkpoint, dict) else 256
    latent_dim = checkpoint.get('latent_dim', 128) if isinstance(checkpoint, dict) else 128
    time_layers = checkpoint.get('time_layers', 4) if isinstance(checkpoint, dict) else 4
    time_heads = checkpoint.get('time_heads', 8) if isinstance(checkpoint, dict) else 8
    dropout = checkpoint.get('dropout', 0.1) if isinstance(checkpoint, dict) else 0.1
    
    model = HumanActionScorer(
        dims_map_raw=dims_map_raw,
        dims_map_diff=dims_map_diff,
        d_model=d_model,
        latent_dim=latent_dim,
        time_layers=time_layers,
        time_heads=time_heads,
        dropout=dropout
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
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
    save_path: str = None
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
            
            seq_embed, frame_embeds, tokens = model(feats)
            
            all_seq_embeds.append(seq_embed.cpu())
            all_frame_embeds.append(frame_embeds.cpu())
            all_cls_names.extend(cls_names)
            all_vid_names.extend(vid_names)
    
    seq_embeds = torch.cat(all_seq_embeds, dim=0)
    frame_embeds = torch.cat(all_frame_embeds, dim=0)
    
    features = {
        'seq_embeds': seq_embeds,
        'frame_embeds': frame_embeds,
        'cls_names': all_cls_names,
        'vid_names': all_vid_names
    }
    
    if save_path:
        torch.save(features, save_path)
        print(f"Saved features to {save_path}")
    
    return features


def compute_frame_difference_scores(features: dict):
    frame_embeds = features['frame_embeds']
    vid_names = features['vid_names']
    
    video_scores = defaultdict(list)
    
    for i, vid_name in enumerate(vid_names):
        video_id = os.path.splitext(vid_name)[0]
        window_frames = frame_embeds[i]
        frames_only = window_frames[1:]
        
        if frames_only.shape[0] < 2:
            continue
        
        frame_diffs = []
        for t in range(1, frames_only.shape[0]):
            diff = torch.norm(frames_only[t] - frames_only[t-1], p=2).item()
            frame_diffs.append(diff)
        
        if frame_diffs:
            avg_window_diff = np.mean(frame_diffs)
            video_scores[video_id].append(avg_window_diff)
    
    video_avg_scores = {}
    for video_id, window_scores in video_scores.items():
        video_avg_scores[video_id] = float(np.mean(window_scores))
    
    return video_avg_scores


def compute_spearman_correlation(model_scores_path: str, human_scores_path: str):
    """
    Compute Spearman correlation between model scores and human scores.
    Uses the same normalization logic as utils.py for consistency.
    """
    def _norm_name(name: str) -> str:
        """Normalize video name to match utils.py logic."""
        stem = os.path.splitext(os.path.basename(name))[0]
        stem = stem.replace("_videos_", "_")
        stem = stem.replace("videos_", "")
        stem = stem.replace("_video_", "_")
        return stem
    
    with open(model_scores_path, 'r') as f:
        model_scores = json.load(f)
    
    with open(human_scores_path, 'r') as f:
        human_scores = json.load(f)
    
    # Normalize model scores by name
    model_by_name = {_norm_name(k): v for k, v in model_scores.items()}
    
    model_values = []
    human_values = []
    matched_videos = []
    
    for human_key, human_data in human_scores.items():
        if "tc" not in human_data:
            continue
        
        human_name_norm = _norm_name(human_key)
        
        # Try exact match first
        if human_name_norm in model_by_name:
            model_values.append(model_by_name[human_name_norm])
            human_values.append(human_data["tc"])
            matched_videos.append((human_name_norm, human_key))
            continue
        
        # Fallback: try partial matching on last parts of name
        human_parts = human_name_norm.split("_")
        for model_name_norm, model_score in model_by_name.items():
            model_parts = model_name_norm.split("_")
            if len(model_parts) >= 2 and len(human_parts) >= 2:
                if (model_parts[-2:] == human_parts[-2:] or 
                    model_parts[-1] == human_parts[-1]):
                    model_values.append(model_score)
                    human_values.append(human_data["tc"])
                    matched_videos.append((model_name_norm, human_key))
                    break
    
    if len(model_values) < 2:
        print(f"Warning: Only {len(model_values)} matched videos. Need at least 2 for correlation.")
        return None, None, matched_videos
    
    model_array = np.array(model_values)
    human_array = np.array(human_values)
    
    correlation, p_value = spearmanr(model_array, human_array)
    
    print(f"\nSpearman Correlation Analysis:")
    print(f"Matched videos: {len(matched_videos)}")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    return correlation, p_value, matched_videos


if __name__ == "__main__":
    generated_meshes_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/generated_meshes"
    model_path = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/model.pt"
    keypoint_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/generated_kps"
    
    print("Loading dataset from generated_meshes...")
    dataset = create_dataset_from_generated_meshes(generated_meshes_dir)
    print(f"Loaded dataset with {len(dataset)} items")
    print(f"Classes: {sorted(dataset.classes)}")
    print(f"Items per class: {[(cls, len(dataset.class_to_items[cls])) for cls in sorted(dataset.classes)]}")
    
    print("Computing modality stats from dataset...")
    stats = compute_stats_from_npz(dataset.items, keypoint_dir=keypoint_dir)
    print(f"Computed stats with {len([f for f in fields(ModalityStats)])} fields")
    
    dims_map_raw, dims_map_diff = infer_dims_from_stats(stats)
    print(f"dims_map_raw: {dims_map_raw}")
    print(f"dims_map_diff: {dims_map_diff}")
    
    print("Loading model...")
    model = load_model(model_path, dims_map_raw, dims_map_diff)
    print("Model loaded successfully")
    
    clip_len = 32
    stride = 8
    print(f"Creating windows (clip_len={clip_len}, stride={stride})...")
    samples = sample_all_windows_npz(dataset, clip_len=clip_len, stride=stride)
    print(f"Created {len(samples)} windows")
    
    window_dataset = WindowDataset(
        samples=samples,
        clip_len=clip_len,
        stats=stats,
        keypoint_dir=keypoint_dir
    )
    
    dataloader = DataLoader(
        window_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda")
    )
    
    print("Extracting window features...")
    features = extract_window_features(
        model=model,
        dataloader=dataloader,
        save_path="window_features.pt"
    )
    
    print(f"Extracted features for {len(features['seq_embeds'])} windows")
    print(f"Sequence embeddings shape: {features['seq_embeds'].shape}")
    print(f"Frame embeddings shape: {features['frame_embeds'].shape}")
    
    print("Computing frame difference scores...")
    video_scores = compute_frame_difference_scores(features)
    
    json_output_path = "video_frame_difference_scores.json"
    with open(json_output_path, 'w') as f:
        json.dump(video_scores, f, indent=2)
    
    print(f"Computed scores for {len(video_scores)} videos")
    print(f"Saved scores to {json_output_path}")
    print(f"Example scores (first 5): {dict(list(video_scores.items())[:5])}")
    
    human_scores_path = "/projectnb/ivc-ml/xthomas/RESEARCH/FINALE/video-gen-evals/human_scores.json"
    if os.path.exists(human_scores_path):
        print("\n" + "="*50)
        correlation, p_value, matched = compute_spearman_correlation(
            json_output_path,
            human_scores_path
        )
        if correlation is not None:
            print(f"\nFinal Result:")
            print(f"Spearman correlation: {correlation:.4f} (p={p_value:.4e})")
    else:
        print(f"\nWarning: Human scores file not found at {human_scores_path}")

