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
from torch import nn
import pandas as pd
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

from scipy.stats import wasserstein_distance

from scipy.spatial.distance import cdist

def precision_recall(real_embeds, gen_embeds, k=3):
    from scipy.spatial.distance import cdist
    dists_real_to_gen = cdist(real_embeds, gen_embeds)  # [N_real, N_gen]
    dists_gen_to_real = dists_real_to_gen.T             # [N_gen, N_real]

    # Recall: For each real, within radius of gen?
    recall_k_dists = np.partition(dists_real_to_gen, k, axis=1)[:, k]
    recall_radius = np.mean(recall_k_dists)
    recall = np.mean((dists_real_to_gen <= recall_radius).any(axis=1))

    # Precision: For each gen, within radius of real?
    precision_k_dists = np.partition(dists_gen_to_real, k, axis=1)[:, k]
    precision_radius = np.mean(precision_k_dists)
    precision = np.mean((dists_gen_to_real <= precision_radius).any(axis=1))

    return precision, recall

from sklearn.metrics.pairwise import rbf_kernel

def compute_mmd(real_embeds, gen_embeds, gamma=None):
    XX = rbf_kernel(real_embeds, real_embeds, gamma=gamma)
    YY = rbf_kernel(gen_embeds, gen_embeds, gamma=gamma)
    XY = rbf_kernel(real_embeds, gen_embeds, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def worst_case_penalty(real_centroid, gen_embeds, percentile=95):
    dists = np.linalg.norm(gen_embeds - real_centroid, axis=1)
    return np.percentile(dists, percentile)

from scipy.spatial.distance import directed_hausdorff

def hausdorff(real_embeds, gen_embeds):
    return max(
        directed_hausdorff(real_embeds, gen_embeds)[0],
        directed_hausdorff(gen_embeds, real_embeds)[0]
    )

def coverage_score(real_embeds, gen_embeds):
    dists = cdist(real_embeds, gen_embeds)  # shape: [N_real, N_gen]
    min_dists = dists.min(axis=1)           # for each real, nearest gen
    return min_dists.mean()                 # lower = better

def compute_wasserstein(real_embeds, gen_embeds):
    # Option 1: Compute 1D W for each dim, then average
    print(f"Computing Wasserstein distance between real ({real_embeds.shape}) and gen ({gen_embeds.shape}) embeddings")
    dists = []
    for d in range(real_embeds.shape[1]):
        dists.append(wasserstein_distance(real_embeds[:, d], gen_embeds[:, d]))
    return np.mean(dists)

def centroidal_outlier_penalty(real_centroid, gen_embeds, alpha=1.0):
    dists = np.linalg.norm(gen_embeds - real_centroid, axis=1)
    return dists.mean() + alpha * dists.std()

from scipy.linalg import sqrtm

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)

def compute_fed(real_embeds, gen_embeds):
    mu_r, sigma_r = real_embeds.mean(0), np.cov(real_embeds, rowvar=False)
    mu_g, sigma_g = gen_embeds.mean(0), np.cov(gen_embeds, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

INPUT_DIM= 1370
def load_video_sequence(video_folder, MESH_DIR, POSE_DIR):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace(MESH_DIR, POSE_DIR)
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
        # elif INPUT_DIM == 1250 + 36:
        vec = np.concatenate([vit_feature, global_orient, body_pose, betas, twod_kp], axis=0)

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

    return enriched_tensor


VID_PATH = {
    'wan21': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5',
    'runway_gen3_alpha': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen3_alpha_videos_5',
    'runway_gen4': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5',
    'hunyuan_360p': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/hunyuan_videos_360p_formatted',
    'opensora_256p':  '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/opensora_videos_256p_formatted',
    'cogvideox':  '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/cogvideox_videos_5'
}

model_class_cons = {} 
model_class_wasserstein = {}
model_class_fid = {}
model_class_penalty = {}
model_class_coverage = {}
model_class_precision = {}
model_class_recall = {}
model_class_mmd = {}
model_class_worst_penalty = {}
model_class_hausdorff = {}
model_class_intra_spread = {}
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

    # ————— load train embeddings & centroids —————
    all_train_embeds = torch.load(f"SAVE_NEW2/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    all_train_labels = torch.load(f"SAVE_NEW2/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    all_train_vid_ids = torch.load(f"SAVE_NEW2/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")

    # get embeddings per class
    real_embeds = defaultdict(list)
    # for emb, lbl in zip(all_train_embeds, all_train_labels):
    #     real_embeds[int(lbl.item())].append(emb.numpy())
    # for k in real_embeds.keys():
    #     real_embeds[k] = np.array(real_embeds[k])
    for idx, cls_name in enumerate(ALL_CLASSES):
        mask_cls = (all_train_labels == idx)
        embeds_cls = all_train_embeds[mask_cls]
        real_embeds[idx] = embeds_cls.numpy()
        
    # compute centroids per class (mean of video means)
    centroid_class = {}
    for cls_name in ALL_CLASSES:
        cls_idx = ALL_CLASSES.index(cls_name)
        embeds_cls = real_embeds[cls_idx]
        if len(embeds_cls) == 0:
            centroid_class[cls_name] = np.zeros((all_train_embeds.shape[1],))
            continue
        centroid_class[cls_name] = embeds_cls.mean(axis=0)

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
    print(" Loading model...")
    model = TemporalTransformer(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)
    print(f" Loading model from temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}.pt")
    model.load_state_dict(torch.load(f"SAVE_NEW2/temporal_transformer_model_window_32_stride_8_valid_window.pt"))
    model.eval()

    # ————— generated dataset & loader —————
    class GeneratedVideoDataset(Dataset):
        def __init__(self, root, classes, window_size=WINDOW_SIZE, stride=STRIDE):
            self.samples, self.labels, self.video_names = [], [], []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for vid in os.listdir(Path(root) / cls):
                    seq = load_video_sequence(Path(root) / cls / vid, MESH_DIR, POSE_DIR)
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
    print("\n Evaluating generated videos...")
    video_results = {}
    video_embeds = defaultdict(list)
    video_labels = {}

    centroids_cpu = {k: v.cpu() for k, v in centroids.items()}


    with torch.no_grad():
        for seqs, labels, vid_names in tqdm(gen_loader):
            seqs = seqs.to(DEVICE)
            lengths = torch.full((seqs.shape[0],), seqs.shape[1], dtype=torch.long, device=DEVICE)
            embs, _ = model(seqs, lengths)
            embs = embs.cpu()
            for emb, lbl, vid in zip(embs, labels, vid_names):
                intra = torch.norm(emb - centroids_cpu[int(lbl.item())]).item()
                inter = [
                    torch.norm(emb - centroids_cpu[k]).item()
                    for k in centroids_cpu.keys()
                    if k != int(lbl.item())
                ]
                inter_mean = np.mean(inter)
                score = float(inter_mean / (inter_mean + intra)) if (intra + inter_mean) != 0 else np.nan
                entry = video_results.setdefault(vid, {"cls": int(lbl), "intra": [], "consistency": []})
                entry["intra"].append(intra)
                entry["consistency"].append(score)
                video_embeds[vid].append(emb.numpy())
                video_labels[vid] = int(lbl)
    
    class_cons = {cls_name: [] for cls_name in ALL_CLASSES}
    for vid, vals in video_results.items():
        cls_idx = vals["cls"]
        cls_name = ALL_CLASSES[cls_idx]
        intra_mean = np.mean(vals["intra"])
        cons_mean = np.mean(vals["consistency"])
        class_cons[cls_name].extend(vals["consistency"])
        
    print(f"\n Results for model {MODEL}:")
    table_data = []
    for cls_name, scores in class_cons.items():
        mean_score = np.mean(scores) if scores else 0
        table_data.append((cls_name, mean_score))
    table_data.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(table_data, headers=["Class", "Mean Consistency Score"], tablefmt="pretty"))

    model_class_cons[MODEL] = {cls_name: np.mean(scores) if scores else 0 for cls_name, scores in class_cons.items()}

    # # get wasserstein distance per class
    # print(f"\n Computing Wasserstein distances for model {MODEL}...")
    # class_wasserstein = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]
    #     w_dist = compute_wasserstein(real_embeds_cls, gen_embeds)
    #     class_wasserstein[cls_name] = w_dist

    # model_class_wasserstein[MODEL] = class_wasserstein

    # # get fid per class
    # print(f"\n Computing FID scores for model {MODEL}...")
    # class_fid = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]
    #     if len(real_embeds_cls) < 2 or len(gen_embeds) < 2:
    #         fid_score = np.nan
    #     else:
    #         fid_score = compute_fed(real_embeds_cls, gen_embeds)
    #     class_fid[cls_name] = fid_score
    
    # model_class_fid[MODEL] = class_fid

    # # get penalty per class
    # print(f"\n Computing Centroidal Outlier Penalties for model {MODEL}...")
    # class_penalty = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_centroid = centroid_class[cls_name]
    #     penalty = centroidal_outlier_penalty(real_centroid, gen_embeds)
    #     class_penalty[cls_name] = penalty
    
    # model_class_penalty[MODEL] = class_penalty

    # # get coverage score per class
    # print(f"\n Computing Coverage Scores for model {MODEL}...")
    # class_coverage = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]
    #     if len(real_embeds_cls) == 0 or len(gen_embeds) == 0:
    #         coverage = np.nan
    #     else:
    #         coverage = coverage_score(real_embeds_cls, gen_embeds)
    #     class_coverage[cls_name] = coverage

    # model_class_coverage[MODEL] = class_coverage

    # # get precision & recall per class
    # print(f"\n Computing Precision & Recall for model {MODEL}...")
    # class_precision = {}
    # class_recall = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]
    #     if len(real_embeds_cls) == 0 or len(gen_embeds) == 0:
    #         prec, rec = np.nan, np.nan
    #     else:
    #         prec, rec = precision_recall(real_embeds_cls, gen_embeds, k=3)
    #     class_precision[cls_name] = prec
    #     class_recall[cls_name] = rec

    # model_class_precision[MODEL] = class_precision
    # model_class_recall[MODEL] = class_recall

    # # get mmd per class
    # print(f"\n Computing MMD scores for model {MODEL}...")
    # class_mmd = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]
    #     if len(real_embeds_cls) == 0 or len(gen_embeds) == 0:
    #         mmd_score = np.nan
    #     else:
    #         mmd_score = compute_mmd(real_embeds_cls, gen_embeds)
    #     class_mmd[cls_name] = mmd_score

    # model_class_mmd[MODEL] = class_mmd

    # # get worst-case penalty per class
    # print(f"\n Computing Worst-Case Penalties for model {MODEL}...")
    # class_worst_penalty = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_centroid = centroid_class[cls_name]
    #     if len(gen_embeds) == 0:
    #         worst_pen = np.nan
    #     else:
    #         worst_pen = worst_case_penalty(real_centroid, gen_embeds, percentile=95)
    #     class_worst_penalty[cls_name] = worst_pen

    # model_class_worst_penalty[MODEL] = class_worst_penalty

    # # get hausdorff per class
    # print(f"\n Computing Hausdorff distances for model {MODEL}...")
    # class_hausdorff = {}
    # for cls_name in ALL_CLASSES:
    #     gen_embeds = []
    #     for vid, vals in video_embeds.items():
    #         if video_labels[vid] == ALL_CLASSES.index(cls_name):
    #             gen_embeds.extend(vals)
    #     gen_embeds = np.array(gen_embeds)
    #     real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]
    #     if len(real_embeds_cls) == 0 or len(gen_embeds) == 0:
    #         haus_score = np.nan
    #     else:
    #         haus_score = hausdorff(real_embeds_cls, gen_embeds)
    #     class_hausdorff[cls_name] = haus_score

    # model_class_hausdorff[MODEL] = class_hausdorff

    # get (intra / intra + spread) score per class
    class_intra_spread = {}
    for cls_name in ALL_CLASSES:
        gen_embeds = []
        for vid, vals in video_embeds.items():
            if video_labels[vid] == ALL_CLASSES.index(cls_name):
                gen_embeds.extend(vals)
        gen_embeds = np.array(gen_embeds)
        real_embeds_cls = real_embeds[ALL_CLASSES.index(cls_name)]

        real_centroid = centroid_class[cls_name]
        gen_intra_dists = np.linalg.norm(gen_embeds - real_centroid, axis=1)
        gen_intra = np.mean(gen_intra_dists) if len(gen_intra_dists) > 0 else np.nan

        gen_spread = np.linalg.norm(gen_embeds - real_centroid, axis=1)
        gen_spread = np.mean(gen_spread) if len(gen_spread) > 0 else np.nan

        if np.isnan(gen_intra) or np.isnan(gen_spread) or (gen_intra + gen_spread) == 0:
            score = np.nan
        else:
            score = gen_intra / (gen_intra + gen_spread)
        class_intra_spread[cls_name] = score

    model_class_intra_spread = {MODEL: class_intra_spread}



print("\n\n================ Final Summary =================")

# print("\n Summary of Precision Scores across Models:")
# summary_table_precision = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21",  "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_precision.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_precision.append(row)

# print(tabulate(summary_table_precision, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Recall Scores across Models:")
# summary_table_recall = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_recall.get(MODEL, {}).get(cls_name, np.nan)
#         if  np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_recall.append(row)

# print(tabulate(summary_table_recall, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of MMD Scores across Models:")
# summary_table_mmd = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_mmd.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_mmd.append(row)

# print(tabulate(summary_table_mmd, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Worst-Case Penalty Scores across Models:")
# summary_table_worst_penalty = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21",  "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_worst_penalty.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_worst_penalty.append(row)

# print(tabulate(summary_table_worst_penalty, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Hausdorff Scores across Models:")
# summary_table_hausdorff = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21",  "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_hausdorff.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_hausdorff.append(row)

# print(tabulate(summary_table_hausdorff, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Coverage Scores across Models:")
# summary_table_coverage = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_coverage.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_coverage.append(row)

# print(tabulate(summary_table_coverage, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Centroidal Outlier Penalties across Models:")
# summary_table_penalty = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_penalty.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_penalty.append(row)

# print(tabulate(summary_table_penalty, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of FID Scores across Models:")
# summary_table_fid = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_fid.get(MODEL, {}).get(cls_name, np.nan)
#         if np.isnan(score):
#             row.append("N/A")
#         else:
#             row.append(f"{score:.4f}")
#     summary_table_fid.append(row)

# print(tabulate(summary_table_fid, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Wasserstein Distances across Models:")
# summary_table_w = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_wasserstein.get(MODEL, {}).get(cls_name, 0)
#         row.append(f"{score:.4f}")
#     summary_table_w.append(row)

# print(tabulate(summary_table_w, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))

# print("\n Summary of Consistency Scores across Models:")
# summary_table = []
# for cls_name in ALL_CLASSES:
#     row = [cls_name]
#     for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
#         score = model_class_cons.get(MODEL, {}).get(cls_name, 0)
#         row.append(f"{score:.4f}")
#     summary_table.append(row)

# print(tabulate(summary_table, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))


print("\n Summary of Intra/Spread Scores across Models:")
summary_table_intra_spread = []
for cls_name in ALL_CLASSES:
    row = [cls_name]
    for MODEL in ["wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"]:
        score = model_class_intra_spread.get(MODEL, {}).get(cls_name, np.nan)
        if np.isnan(score):
            row.append("N/A")
        else:
            row.append(f"{score:.4f}")
    summary_table_intra_spread.append(row)

print(tabulate(summary_table_intra_spread, headers=["Class", "wan21", "runway_gen4", "hunyuan", "opensora", "cogvideox"], tablefmt="pretty"))