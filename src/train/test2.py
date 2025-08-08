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
from sklearn.manifold import TSNE
from collections import defaultdict
import torch.nn.functional as F

from train import  extract_windows
from train2 import (
    TemporalTransformer,
    # load_video_sequence,
    # extract_windows,
    collate_fn,
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)
from tabulate import tabulate

INPUT_DIM= 1370
POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN4"
def load_video_sequence(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos", POSE_DIR)
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

        # print(np.array(params["token_out"]).shape, np.array(params["global_orient"]).shape, np.array(params["body_pose"]).shape, np.array(params["betas"]).shape)

        vit_feature   = np.array(params["token_out"])[0].flatten()
        global_orient = np.array(params["global_orient"])[0].flatten()
        body_pose     = np.array(params["body_pose"])[0].flatten()
        betas         = np.array(params["betas"])[0].flatten()

        twod_point_path = twod_points_paths[idx]
        twod_kp = np.load(twod_point_path).flatten()
        twod_kp = twod_kp[:120]  # Ensure 36 keypoints

        # # Normalize each part
        vit_feature   /= np.linalg.norm(vit_feature) + 1e-8
        global_orient /= np.linalg.norm(global_orient) + 1e-8
        body_pose     /= np.linalg.norm(body_pose) + 1e-8
        betas         /= np.linalg.norm(betas) + 1e-8
        twod_kp       /= np.linalg.norm(twod_kp) + 1e-8
        # twod_kp = twod_kp[:36]  # Ensure 120 keypoints

        # if INPUT_DIM == 1250:
        #     vec = np.concatenate([vit_feature, global_orient, body_pose, betas], axis=0)
        #     vec = vec / np.linalg.norm(vec) + 1e-8
        #     if vec.shape[0] != 1250:
        #         continue
        # elif INPUT_DIM == 226:
        #     vec = np.concatenate([global_orient, body_pose, betas], axis=0)
        #     vec = vec / np.linalg.norm(vec) + 1e-8
        #     if vec.shape[0] != 226:
        #         continue
        # elif INPUT_DIM == 1250 + 36:
        vec = np.concatenate([vit_feature, global_orient, body_pose, betas, twod_kp], axis=0)
        # vec = vec / np.linalg.norm(vec) + 1e-8
        # if vec.shape[0] != 1250 + 36:
        #     continue

        frame_vecs.append(torch.tensor(vec, dtype=torch.float32))
        # except:
        #     continue

    if len(frame_vecs) < 2:
        return None

    # [T, 1250]
    frame_tensor = torch.stack(frame_vecs, dim=0)

    # Compute motion vectors (frame-to-frame deltas)
    motion_vecs = frame_tensor[1:] - frame_tensor[:-1]  # [T-1, 1250])
    motion_vecs = torch.cat([torch.zeros(1, INPUT_DIM), motion_vecs], dim=0)  # [T, 1250]


    # Concatenate original + motion
    enriched_tensor = torch.cat([frame_tensor, motion_vecs], dim=1)  # [T, 2500] # 2500 = 1250 * 2
    # print(enriched_tensor.shape)
    # exit()

    # # return enriched_tensor
    # if INPUT_DIM == (1250 + 36) * 2:
    #     return enriched_tensor
    # return frame_tensor
    return enriched_tensor


VID_PATH = {
    'wan21': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5',
    'runway_gen3_alpha': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen3_alpha_videos_5',
    'runway_gen4': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5',
}

for MODEL in ["runway_gen4"]:
    GEN_ROOT = f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_{MODEL}_videos"
    if "cogvideox" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "opensora" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "hunyuan" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "runway" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "wan21" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    else:
        WINDOW_SIZE = 32
        STRIDE = 8
    BATCH_SIZE = 64

    # ————— load train embeddings & centroids —————
    all_train_embeds = torch.load(f"SAVE_NEW2/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    all_train_labels = torch.load(f"SAVE_NEW2/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    all_train_vid_ids = torch.load(f"SAVE_NEW2/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    # with open(f"SAVE_NEW2/centroids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pkl", "rb") as f:
    #     centroids = pickle.load(f)
    # for k in centroids:
    #     centroids[k] = centroids[k].to(DEVICE)
    # Compute class centroids
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

    # ————— load model —————
    print("✅ Loading model...")
    model = TemporalTransformer(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)
    print(f"✅ Loading model from temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}.pt")
    model.load_state_dict(torch.load(f"SAVE_NEW2/temporal_transformer_model_window_32_stride_8_valid_window.pt"))
    model.eval()

    # ————— generated dataset & loader —————
    class GeneratedVideoDataset(Dataset):
        def __init__(self, root, classes, window_size=WINDOW_SIZE, stride=STRIDE):
            self.samples, self.labels, self.video_names = [], [], []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for vid in os.listdir(Path(root) / cls):
                    seq = load_video_sequence(Path(root) / cls / vid)
                    if seq is None:
                        continue
                    wins = extract_windows(seq, window_size, stride)
                    self.samples.extend(wins)
                    self.labels.extend([self.class_to_idx[cls]] * len(wins))
                    self.video_names.extend([vid] * len(wins))
            print(f"✅ Loaded {len(self.samples)} windows from generated videos")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx], self.video_names[idx]

    gen_dataset = GeneratedVideoDataset(GEN_ROOT, ALL_CLASSES)
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

    # ————— evaluate & collect per-window scores & embeddings —————
    print("\n✅ Evaluating generated videos...")
    video_results = {}
    video_embeds = defaultdict(list)
    video_labels = {}

    with torch.no_grad():
        for seqs, labels, vid_names in tqdm(gen_loader):
            seqs = seqs.to(DEVICE)
            lengths = torch.full((seqs.shape[0],), seqs.shape[1], dtype=torch.long, device=DEVICE)
            embs, _ = model(seqs, lengths)
            embs = embs.cpu()

            centroids_norm = {k: v / torch.norm(v) for k, v in centroids.items()}
            # move centroids to cpu
            centroids_norm = {k: v.cpu() for k, v in centroids_norm.items()}

            # Now, inside your per-embedding loop:
            for emb, lbl, vid in zip(embs, labels, vid_names):
                emb = emb / torch.norm(emb)
                intra = torch.norm(emb - centroids_norm[int(lbl)]).item()
                inter = [
                    torch.norm(emb - centroids_norm[c]).item()
                    for c in centroids_norm
                    if c != int(lbl)
                ]
                # inter_mean = np.mean(inter)
                # score = float(inter_mean / (inter_mean + intra)) if (intra + inter_mean) != 0 else np.nan
                inter_min = np.min(inter) if len(inter) > 0 else 0.0
                # The margin-based score: higher if intra << inter_min, lower if intra > inter_min
                score = float((inter_min - intra) / (inter_min + intra + 1e-8))

                entry = video_results.setdefault(vid, {"cls": int(lbl), "intra": [], "consistency": []})
                entry["intra"].append(intra)
                entry["consistency"].append(score)

                video_embeds[vid].append(emb)
                video_labels[vid] = int(lbl)

    # ————— rank by outlier-aware (weighted) averaging —————
    video_scores = []
    for vid, vals in video_results.items():
        mean_intra = np.max(vals["intra"])

        scores = np.array(vals["consistency"])
        mu, sigma = scores.mean(), scores.std() if scores.std() > 0 else 1e-6
        weights = np.exp(-((scores - mu) ** 2) / (2 * sigma ** 2))
        weights /= weights.sum()
        weighted_consistency = float((weights * scores).sum())
        # scores = np.array(vals["consistency"])
        # scores_tensor = torch.tensor(scores)
        # weights = torch.softmax(scores_tensor, dim=0).numpy()
        # weighted_consistency = float((weights * scores).sum())

        # video_scores.append((vid, ALL_CLASSES[vals["cls"]], mean_intra, weighted_consistency))

        # --- Compute spread: mean distance to video-mean ---
        emb_stack = torch.stack(video_embeds[vid])  # [N, D]
        spread = torch.norm(emb_stack - emb_stack.mean(dim=0, keepdim=True), dim=1).mean().item()
        # (or use torch.pdist(emb_stack).mean().item() if you prefer mean pairwise)

        # Combine: can tune beta to weight spread penalty
        beta = 1.0
        penalized_score = weighted_consistency - beta * spread  # *higher is better* if you want to reward tightness

        video_scores.append((vid, ALL_CLASSES[vals["cls"]], mean_intra, weighted_consistency, spread, penalized_score))

    video_scores.sort(key=lambda x: -x[3])

    print("\n✅ Generated Video Ranking (Best to Worst by Weighted Consistency):")
    for rank, (vid, cls_name, intra, cons, spread, penalized_score) in enumerate(video_scores, 1):
        print(f"{rank}. {vid} | Class: {cls_name} | Intra: {intra:.4f} | Weighted Consistency: {cons:.4f} | Spread: {spread:.4f} | Penalized Score: {penalized_score:.4f}")
        # print(tabulate(video_scores, headers=["Rank", "Video", "Class", "Intra", "Weighted Consistency"], tablefmt="grid"))

    video_scores_json = [
        {
            "video": str(vid),
            "class": str(cls_name),
            "mean_intra": float(intra),
            "weighted_consistency": float(cons),
            "spread": float(spread),
            "penalized_score": float(penalized_score),
        }
        for (vid, cls_name, intra, cons, spread, penalized_score) in video_scores
    ]

    with open(f"SAVE_NEW2/jsons/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.json", "w") as f:
        json.dump(video_scores_json, f)
    print(f"✅ Saved as SAVE_NEW2/jsons/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.json")

    # ————— prepare per-video averaged embeddings for t-SNE —————
    vid_names = list(video_embeds.keys())
    vid_embeds_list = []
    for v in vid_names:
        emb_stack = torch.stack(video_embeds[v])               # [N, D]
        centroid = centroids[video_labels[v]].cpu()            # [D]
        dists = torch.norm(emb_stack - centroid, dim=1)        # [N]
        weights = dists / (dists.sum() + 1e-8)                 # linear weighting
        weighted_avg = (weights.unsqueeze(1) * emb_stack).sum(dim=0)
        vid_embeds_list.append(weighted_avg.numpy())


    vid_embeds = np.stack(vid_embeds_list)
    vid_classes = np.array([video_labels[v] for v in vid_names])
    centroid_mat = torch.stack([centroids[c].cpu() for c in sorted(centroids)]).numpy()

    # ————— t-SNE projection —————
    tsne = PCA(n_components=2, random_state=42)
    combined = np.concatenate([vid_embeds, centroid_mat], axis=0)
    # normalize combined
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    proj = tsne.fit_transform(combined)
    projected_vid_embeds = proj[: len(vid_embeds)]
    projected_centroids = proj[len(vid_embeds) :]

    # ————— plot —————
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
        # annotate each point with its video name
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            vid_idx = np.where(mask)[0][i]
            vid_name = vid_names[vid_idx]
            plt.text(x, y, vid_name, fontsize=5, ha="center", va="center", alpha=0.6)

    for i, (x, y) in enumerate(projected_centroids):
        plt.scatter(x, y, edgecolors="k", s=200, marker="X", linewidths=2)
        plt.text(x, y + 0.05, ALL_CLASSES[i], fontsize=9, ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.8))

    plt.title("Per-Video Averaged Embeddings + Centroids (t-SNE Projection)")
    # plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"SAVE_NEW2/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png", dpi=200)
    print(f"✅ Saved as gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")
    plt.close()

    # ————— plot —————
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
        # annotate each point with its video name
        # for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        #     vid_idx = np.where(mask)[0][i]
        #     vid_name = vid_names[vid_idx]
        #     plt.text(x, y, vid_name, fontsize=5, ha="center", va="center", alpha=0.6)

    for i, (x, y) in enumerate(projected_centroids):
        plt.scatter(x, y, edgecolors="k", s=200, marker="X", linewidths=2)
        plt.text(x, y + 0.05, ALL_CLASSES[i], fontsize=9, ha='center', va='bottom', color='black', bbox=dict(facecolor='white', alpha=0.8))

    plt.title("Per-Video Averaged Embeddings + Centroids (t-SNE Projection)")
    # plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"SAVE_NEW2/gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_no_annot_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png", dpi=200)
    print(f"✅ Saved as gen_{GEN_ROOT.split('/')[-1]}_embeddings_centroids_per_video_no_annot_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.png")

    # ————— Group by class and get Top-3 / Bottom-3 —————
    from collections import defaultdict

    class_video_ranking = defaultdict(list)
    for vid, cls_name, intra, cons, spread, penalized_score in video_scores:
        class_video_ranking[cls_name].append((vid, intra, cons, spread, penalized_score))

    top_bottom_summary = []

    print("\n📊 Top 3 & Worst 3 Videos per Class (by Weighted Consistency):")
    for cls_name, videos in class_video_ranking.items():
        # Sort by consistency instead of penalized score
        videos.sort(key=lambda x: x[1])  # x[2] is consistency
        top3 = videos[:3]
        worst3 = videos[-3:]

        print(f"\n🔹 Class: {cls_name}")
        print("Top 3:")
        for rank, (vid, intra, cons, spread, penalized_score) in enumerate(top3, 1):
            print(f"  {rank}. {vid} | Consistency: {cons:.4f} | Intra: {intra:.4f} | Spread: {spread:.4f} | Penalized Score: {penalized_score:.4f}")
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
        for rank, (vid, intra, cons, spread, penalized_score) in enumerate(worst3[::-1], 1):
            print(f"  {rank}. {vid} | Consistency: {cons:.4f} | Intra: {intra:.4f} | Spread: {spread:.4f} | Penalized Score: {penalized_score:.4f}")
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

    # ————— Save summary to JSON —————
    summary_path = f"SAVE_NEW2/jsons/gen_{GEN_ROOT.split('/')[-1]}_top_bottom_by_consistency_per_class_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.json"
    with open(summary_path, "w") as f:
        json.dump(top_bottom_summary, f, indent=2)
    print(f"\n✅ Saved Top/Bottom-by-Consistency summary: {summary_path}")


    print("SAVE VIDES sorted into a new DIR")
    SAVE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/DEL/RUNWAY_GEN4"
    for cls_name, videos in class_video_ranking.items():
        # Sort by consistency instead of penalized score
        videos.sort(key=lambda x: x[1])  # x[2] is consistency

        for rank, (vid, intra, cons, spread, penalized_score) in enumerate(videos[:3], 1):
            src_path = Path(VID_PATH[MODEL]) / cls_name / vid / "video.mp4"
            dst_dir = Path(SAVE_DIR) / cls_name 
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / f"top_{rank}_{vid}_score_{intra:.3f}_spread_{spread:.3f}.mp4"
            os.system(f"cp {src_path} {dst_path}")
            print(f"Copied {src_path} to {dst_path}")

        videos.sort(key=lambda x: x[1], reverse=True)  # Sort by intra distance for worst
        for rank, (vid, intra, cons, spread, penalized_score) in enumerate(videos[:3], 1):
            src_path = Path(VID_PATH[MODEL]) / cls_name / vid / "video.mp4"
            dst_dir = Path(SAVE_DIR) / cls_name 
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = dst_dir / f"worst_{rank}_{vid}_score_{intra:.3f}_spread_{spread:.3f}.mp4"
            os.system(f"cp {src_path} {dst_path}")
            print(f"Copied {src_path} to {dst_path}")

    print("✅ Evaluation complete!")