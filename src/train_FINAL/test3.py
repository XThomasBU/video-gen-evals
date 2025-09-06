import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
import pandas as pd
from save_data import extract_windows
from train import (
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)
from models import TemporalTransformerV2Plus
from utils import *
from tabulate import tabulate

INPUT_DIM = 1370

def _axis_angle_to_matrix(a: torch.Tensor) -> torch.Tensor:
    theta = a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = a / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    O = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([O,   -kz,  ky], dim=-1),
        torch.stack([kz,   O,  -kx], dim=-1),
        torch.stack([-ky,  kx,   O], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(a.shape[:-1] + (3, 3))
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return I + s * K + (1.0 - c) * (K @ K)

def _log_so3(R: torch.Tensor) -> torch.Tensor:
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
    v = F.normalize(vit, dim=-1)
    v_prev = torch.cat([v[:1], v[:-1]], dim=0)
    return v - v_prev

def _rot_axisangle_delta(aa: torch.Tensor) -> torch.Tensor:
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
    T = Rflat.shape[0]
    R = Rflat.view(T, 3, 3)
    R0 = torch.cat([R[:1], R[:-1]], dim=0)
    Rrel = torch.matmul(R0.transpose(-1, -2), R)
    return Rrel.reshape(T, 9)

def _procrustes_kp_delta(kp: torch.Tensor) -> torch.Tensor:
    T, D = kp.shape
    pts = kp.view(T, -1, 2)
    pts_c = pts - pts.mean(dim=1, keepdim=True)
    s = torch.linalg.norm(pts_c, dim=(1, 2), keepdim=True).clamp_min(1e-6)
    pts_n = pts_c / s
    prev = torch.cat([pts_n[:1], pts_n[:-1]], dim=0)
    return (pts_n - prev).reshape(T, D)

def _betas_delta(betas: torch.Tensor, ema: float = 0.9, max_abs: float = 0.1) -> torch.Tensor:
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
    vit   = torch.stack(vit_list, dim=0)
    go    = torch.stack(go_list, dim=0)
    pose  = torch.stack(pose_list, dim=0)
    betas = torch.stack(betas_list, dim=0)
    kp2d  = torch.stack(kp_list, dim=0)
    def z(x, mean, std):
        return (x - torch.tensor(mean, dtype=x.dtype)) / (torch.tensor(std, dtype=x.dtype) + 1e-8)
    vit_raw   = z(vit,   stats['vit_mean'],           stats['vit_std'])
    go_raw    = z(go,    stats['global_orient_mean'], stats['global_orient_std'])
    pose_raw  = z(pose,  stats['body_pose_mean'],     stats['body_pose_std'])
    betas_raw = z(betas, stats['betas_mean'],         stats['betas_std'])
    kp_raw    = z(kp2d,  stats['twod_kp_mean'],       stats['twod_kp_std'])
    raw = torch.cat([vit_raw, go_raw, pose_raw, betas_raw, kp_raw], dim=-1)
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
    return enriched

VID_PATH = {
    'wan21': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5',
    'runway_gen3_alpha': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen3_alpha_videos_5',
    'runway_gen4': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5',
    'hunyuan_360p': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/hunyuan_videos_360p_formatted',
    'opensora_256p':  '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/opensora_videos_256p_formatted',
    'cogvideox':  '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/cogvideox_videos_5'
}

for MODEL in ["wan21", "runway_gen4", "hunyuan_360p", "opensora_256p", "cogvideox"]:
    GEN_ROOT = f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_{MODEL}_videos"
    if "cogvideox" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_COGVIDEOX"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos"
    elif "opensora" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_OPENSORA_256p"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_opensora_256p_videos"
    elif "hunyuan" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_HUNYUAN_360p"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_hunyuan_360p_videos"
    elif "runway" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        if MODEL == "runway_gen3_alpha":
            POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN3_ALPHA"
            MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen3_alpha_videos"
        elif MODEL == "runway_gen4":
            POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN4"
            MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos"
    elif "wan21" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_WAN21"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_wan21_videos"
    else:
        WINDOW_SIZE = 32
        STRIDE = 8
    BATCH_SIZE = 64

    all_train_embeds = torch.load(f"SAVE/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    all_train_labels = torch.load(f"SAVE/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    all_train_vid_ids = torch.load(f"SAVE/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt", weights_only=False)

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

    # ---- NEW: class-conditional precision matrices on unnormalized train embeddings
    class_means = {}
    class_precisions = {}
    lam = 1e-3
    with torch.no_grad():
        for cls in torch.unique(all_train_labels):
            c = int(cls.item())
            X = all_train_embeds[all_train_labels == cls].float().cpu()        # [N,D]
            mu = X.mean(dim=0, keepdim=True)                                   # [1,D]
            Xc = X - mu
            # cov with ddof=1 if N>1
            if Xc.shape[0] > 1:
                C = (Xc.t() @ Xc) / (Xc.shape[0] - 1)
            else:
                D = Xc.shape[1]
                C = torch.eye(D)
            D = C.shape[0]
            C = C + lam * torch.eye(D)
            P = torch.linalg.inv(C)                                           # precision
            class_means[c] = mu.squeeze(0).contiguous()
            class_precisions[c] = P.contiguous()

    print(" Loading model...")
    model = TemporalTransformerV2Plus(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)
    print(f" Loading model from temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_NO_ENT.pt")
    model.load_state_dict(torch.load(f"SAVE/temporal_transformer_model_window_32_stride_8_valid_window_NO_ENT.pt"))
    model.eval()

    class GeneratedVideoDataset(Dataset):
        def __init__(self, root, classes, window_size=WINDOW_SIZE, stride=STRIDE):
            self.samples, self.labels, self.video_names = [], [], []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for vid in os.listdir(Path(root) / cls):
                    seq = load_video_sequence(Path(root) / cls / vid, MESH_DIR, POSE_DIR, "SAVE")
                    if seq is None:
                        continue
                    wins = extract_windows(seq, window_size, stride)
                    self.samples.extend(wins)
                    self.labels.extend([self.class_to_idx[cls]] * len(wins))
                    self.video_names.extend([vid] * len(wins))
            print(f" Loaded {len(self.samples)} windows from generated videos")
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx], self.video_names[idx]

    if os.path.exists(f"SAVE_TEST/gen_dataset_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_MODEL_{MODEL}.pt"):
        gen_dataset = torch.load(f"SAVE_TEST/gen_dataset_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_MODEL_{MODEL}.pt", weights_only=False)
    else:
        gen_dataset = GeneratedVideoDataset(GEN_ROOT, ALL_CLASSES)
        torch.save(gen_dataset, f"SAVE_TEST/gen_dataset_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_MODEL_{MODEL}.pt")
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: (
            torch.stack([i[0] for i in x]),
            torch.tensor([i[1] for i in x]),
            [i[2] for i in x],
        ),
    )

    print("\n Evaluating generated videos...")
    video_results = {}
    video_embeds = defaultdict(list)
    video_labels = {}

    def maha_dist_sq(x: torch.Tensor, mu: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        d = x - mu
        return (d @ P @ d)

    with torch.no_grad():
        centroids_cpu = {k: v.cpu() for k, v in centroids.items()}
        for seqs, labels, vid_names in tqdm(gen_loader):
            seqs = seqs.to(DEVICE)
            lengths = torch.full((seqs.shape[0],), seqs.shape[1], dtype=torch.long, device=DEVICE)
            embs, _, _ = model(seqs, lengths)
            embs = embs.cpu()  # UNNORMALIZED embeddings now carry magnitude

            for emb, lbl, vid in zip(embs, labels, vid_names):
                cls = int(lbl)
                intra = float(torch.sqrt(maha_dist_sq(emb, centroids_cpu[cls], class_precisions[cls])))
                inter = []
                for c in centroids_cpu:
                    if c == cls: 
                        continue
                    inter_d = float(torch.sqrt(maha_dist_sq(emb, centroids_cpu[c], class_precisions[c])))
                    inter.append(inter_d)
                inter_min = np.min(inter) if len(inter) > 0 else 0.0
                score = float((inter_min - intra) / (inter_min + intra + 1e-8))

                entry = video_results.setdefault(vid, {"cls": cls, "intra": [], "consistency": []})
                entry["intra"].append(intra)
                entry["consistency"].append(score)

                video_embeds[vid].append(emb)
                video_labels[vid] = cls

    video_scores = []
    real_embeds = {}
    for i, cls in enumerate(ALL_CLASSES):
        mask = (all_train_labels == i)
        real_embeds[i] = all_train_embeds[mask].numpy()
    for vid, vals in video_results.items():
        mean_intra = np.mean(vals["intra"])
        max_intra = np.max(vals["intra"])
        scores = np.array(vals["consistency"])
        mu_s, sigma_s = scores.mean(), scores.std() if scores.std() > 0 else 1e-6
        weights = np.exp(-((scores - mu_s) ** 2) / (2 * sigma_s ** 2))
        weights /= weights.sum()
        weighted_consistency = float((weights * scores).sum())

        emb_stack = torch.stack(video_embeds[vid])
        video_mean_emb = emb_stack.mean(dim=0, keepdim=True)

        cls_idx = vals["cls"]
        P_cls = class_precisions[cls_idx].cpu().numpy()
        real_cls = real_embeds[cls_idx]
        vm = video_mean_emb.cpu().numpy().squeeze()

        diffs = real_cls - vm[None, :]
        dists_sq = np.einsum('ni,ij,nj->n', diffs, P_cls, diffs)
        dists = np.sqrt(np.maximum(dists_sq, 0.0))
        energy = float(dists.mean())
        percentile_penalty = float(np.percentile(dists, 95))
        hausdorff_score = float(dists.max())

        spread_sq = (emb_stack.cpu().numpy() - vm[None, :])
        spread_sq = np.einsum('ni,ij,nj->n', spread_sq, P_cls, spread_sq)
        spread = float(np.sqrt(np.maximum(spread_sq, 0.0)).mean())

        beta = 1.0
        gamma = 1.0
        delta = 1.0
        penalized_score = float(
            weighted_consistency
            - beta * spread
            - gamma * energy
            - delta * percentile_penalty
        )

        video_scores.append((
            vid, ALL_CLASSES[cls_idx], mean_intra, weighted_consistency, spread, energy,
            percentile_penalty, hausdorff_score, penalized_score, max_intra
        ))

    video_scores.sort(key=lambda x: x[2])

    print("\n Generated Video Ranking (Best to Worst by Weighted Consistency):")
    for rank, (vid, cls_name, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra) in enumerate(video_scores, 1):
        print(f"{rank}. {vid} | Class: {cls_name} | Intra: {intra:.4f} | Weighted Consistency: {cons:.4f} | Spread: {spread:.4f} | Energy: {energy:.4f} | 95th Perc Penalty: {percentile_penalty:.4f} | Hausdorff: {hausdorff_score:.4f} | Penalized: {penalized_score:.4f} | Max Intra: {max_intra:.4f}")

    video_scores_json = [
        {
            "video": str(vid),
            "class": str(cls_name),
            "mean_intra": float(intra),
            "weighted_consistency": float(cons),
            "spread": float(spread),
            "energy": float(energy),
            "percentile_penalty": float(percentile_penalty),
            "hausdorff_score": float(hausdorff_score),
            "penalized_score": float(penalized_score),
            "max_intra": float(max_intra)
        }
        for (vid, cls_name, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra) in video_scores
    ]

    with open(f"SAVE_TEST/jsons/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.json", "w") as f:
        json.dump(video_scores_json, f)
    print(f" Saved as SAVE_TEST/jsons/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.json")

    vid_names = list(video_embeds.keys())
    vid_embeds_list = []
    for v in vid_names:
        emb_stack = torch.stack(video_embeds[v])
        centroid = centroids[video_labels[v]].cpu()
        dists = torch.norm(emb_stack - centroid, dim=1)
        weights = dists / (dists.sum() + 1e-8)
        weighted_avg = (weights.unsqueeze(1) * emb_stack).sum(dim=0)
        vid_embeds_list.append(weighted_avg.numpy())

    vid_embeds = np.stack(vid_embeds_list)
    vid_classes = np.array([video_labels[v] for v in vid_names])
    centroid_mat = torch.stack([centroids[c].cpu() for c in sorted(centroids)]).numpy()

    tsne = PCA(n_components=2, random_state=42)
    combined = np.concatenate([vid_embeds, centroid_mat], axis=0)
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    proj = tsne.fit_transform(combined)
    projected_vid_embeds = proj[: len(vid_embeds)]
    projected_centroids = proj[len(vid_embeds) :]

    colors = plt.cm.get_cmap("tab10", len(ALL_CLASSES))
    plt.figure(figsize=(10, 8))
    for cls in range(len(ALL_CLASSES)):
        mask = vid_classes == cls
        x_coords = projected_vid_embeds[mask, 0]
        y_coords = projected_vid_embeds[mask, 1]
        plt.scatter(
            projected_vid_embeds[mask, 0],
            projected_vid_embeds[mask, 1],
            s=40,
            color=colors(cls),
            label=f"{ALL_CLASSES[cls]} (gen)",
            alpha=0.9,
        )
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            vid_idx = np.where(mask)[0][i]
            vid_name = vid_names[vid_idx]
            plt.text(x, y, vid_name, fontsize=5, ha="center", va="center", alpha=0.6)
    for i, (x, y) in enumerate(projected_centroids):
        plt.scatter(x, y, edgecolors="k", s=200, marker="X", linewidths=2)
        plt.text(x, y + 0.05, ALL_CLASSES[i], fontsize=9, ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.8))
    plt.title("Per-Video Averaged Embeddings + Centroids (t-SNE Projection)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"SAVE_TEST/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png", dpi=200)
    print(f" Saved as gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")
    plt.close()

    colors = plt.cm.get_cmap("tab10", len(ALL_CLASSES))
    plt.figure(figsize=(10, 8))
    for cls in range(len(ALL_CLASSES)):
        mask = vid_classes == cls
        plt.scatter(
            projected_vid_embeds[mask, 0],
            projected_vid_embeds[mask, 1],
            s=40,
            color=colors(cls),
            label=f"{ALL_CLASSES[cls]} (gen)",
            alpha=0.9,
        )
    for i, (x, y) in enumerate(projected_centroids):
        plt.scatter(x, y, edgecolors="k", s=200, marker="X", linewidths=2)
        plt.text(x, y + 0.05, ALL_CLASSES[i], fontsize=9, ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.8))
    plt.title("Per-Video Averaged Embeddings + Centroids (t-SNE Projection)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"SAVE_TEST/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_no_annot_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png", dpi=200)
    print(f" Saved as gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_no_annot_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.png")

    class_video_ranking = defaultdict(list)
    for vid, cls_name, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra in video_scores:
        class_video_ranking[cls_name].append((vid, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra))

    top_bottom_summary = []
    print("\n📊 Top 3 & Worst 3 Videos per Class (by Weighted Consistency):")
    for cls_name, videos in class_video_ranking.items():
        videos.sort(key=lambda x: x[1])
        top3 = videos[:3]
        worst3 = videos[-3:]
        print(f"\n🔹 Class: {cls_name}")
        print("Top 3:")
        for rank, (vid, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra) in enumerate(top3, 1):
            print(f"  {rank}. {vid} | Consistency: {cons:.4f} | Intra: {intra:.4f} | Spread: {spread:.4f} | Energy: {energy:.4f} | Percentile Penalty: {percentile_penalty:.4f} | Hausdorff: {hausdorff_score:.4f} | Penalized: {penalized_score:.4f} | Max Intra: {max_intra:.4f}")
            top_bottom_summary.append({
                "class": cls_name,
                "type": "top",
                "video": vid,
                "rank": rank,
                "consistency": cons,
                "intra": intra,
                "spread": spread,
                "penalized_score": penalized_score
            })
        print("Worst 3:")
        for rank, (vid, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra) in enumerate(worst3[::-1], 1):
            print(f"  {rank}. {vid} | Consistency: {cons:.4f} | Intra: {intra:.4f} | Spread: {spread:.4f} | Energy: {energy:.4f} | Percentile Penalty: {percentile_penalty:.4f} | Hausdorff: {hausdorff_score:.4f} | Penalized: {penalized_score:.4f} | Max Intra: {max_intra:.4f}")
            top_bottom_summary.append({
                "class": cls_name,
                "type": "worst",
                "video": vid,
                "rank": rank,
                "consistency": cons,
                "intra": intra,
                "spread": spread,
                "penalized_score": penalized_score
            })

    summary_path = f"SAVE_TEST/jsons/gen_{GEN_ROOT.split('/')[-1]}_top_bottom_by_consistency_per_class_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.json"
    with open(summary_path, "w") as f:
        json.dump(top_bottom_summary, f, indent=2, default=float)
    print(f"\n Saved Top/Bottom-by-Consistency summary: {summary_path}")

    print("SAVE VIDEOS sorted into a new DIR")
    SAVE_DIR = f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train_FINAL2backup/SAVE_TEST/models/{MODEL}_NO_ENT"
    for cls_name, videos in class_video_ranking.items():
        videos.sort(key=lambda x: x[1])
        for rank, (vid, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra) in enumerate(videos[:3], 1):
            src_path = Path(VID_PATH[MODEL]) / cls_name / vid / "video.mp4"
            dst_dir = Path(SAVE_DIR) / cls_name
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / f"top_{rank}_{vid}_score_{intra:.3f}_spread_{spread:.3f}.mp4"
            os.system(f"cp {src_path} {dst_path}")
            print(f"Copied {src_path} to {dst_path}")
        videos.sort(key=lambda x: x[1], reverse=True)
        for rank, (vid, intra, cons, spread, energy, percentile_penalty, hausdorff_score, penalized_score, max_intra) in enumerate(videos[:3], 1):
            src_path = Path(VID_PATH[MODEL]) / cls_name / vid / "video.mp4"
            dst_dir = Path(SAVE_DIR) / cls_name
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / f"worst_{rank}_{vid}_score_{intra:.3f}_spread_{spread:.3f}.mp4"
            os.system(f"cp {src_path} {dst_path}")
            print(f"Copied {src_path} to {dst_path}")

    print(" Evaluation complete!")

    class_summary = []
    for cls_name, videos in class_video_ranking.items():
        n = len(videos)
        mean_consistency = np.mean([v[2] for v in videos])
        mean_intra = np.mean([v[1] for v in videos])
        mean_spread = np.mean([v[3] for v in videos])
        mean_penalized = np.mean([v[7] for v in videos])
        mean_energy = np.mean([v[4] for v in videos])
        mean_percentile_penalty = np.mean([v[5] for v in videos])
        mean_hausdorff = np.mean([v[6] for v in videos])
        max_intra = np.max([v[8] for v in videos])
        class_summary.append({
            "class": cls_name,
            "num_videos": n,
            "mean_consistency": mean_consistency,
            "mean_intra": mean_intra,
            "mean_spread": mean_spread,
            "mean_penalized_score": mean_penalized,
            "mean_energy": mean_energy,
            "mean_percentile_penalty": mean_percentile_penalty,
            "mean_hausdorff": mean_hausdorff,
            "max_intra": max_intra
        })
    df_summary = pd.DataFrame(class_summary)
    print("\n📊 Per-Class Average Scores:")
    print(df_summary)