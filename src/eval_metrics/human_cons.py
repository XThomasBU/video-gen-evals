import os
import json
import numpy as np
import argparse
import tqdm
from pathlib import Path
from scipy.interpolate import splprep, splev
import torch
import matplotlib.pyplot as plt
import ot
from collections import Counter
import cv2
import imageio
from diffusers.utils import export_to_video
import shutil

def chamfer_distance_single(points1, points2):
    points1 = torch.from_numpy(points1).unsqueeze(0).float()
    points2 = torch.from_numpy(points2).unsqueeze(0).float()
    dist1 = torch.cdist(points1, points2).squeeze(0)
    cd_forward = dist1.min(dim=1)[0].mean()
    cd_backward = dist1.min(dim=0)[0].mean()
    return ((cd_forward + cd_backward) / 2.0).item()

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_pelvis_curvature(pelvis_array):
    t = np.linspace(0, 1, len(pelvis_array))
    tck, _ = splprep(pelvis_array.T, u=t, s=0)
    deriv1 = np.array(splev(t, tck, der=1)).T
    deriv2 = np.array(splev(t, tck, der=2)).T
    numerator = np.linalg.norm(np.cross(deriv1, deriv2), axis=1)
    denominator = np.linalg.norm(deriv1, axis=1) ** 3 + 1e-8
    return (numerator / denominator).mean()

def compute_consistency_metrics(json_folder, exp_consistency=False):
    json_folder = Path(json_folder)
    json_files = sorted(list(json_folder.glob("*.json")))
    betas_list, body_pose_list, global_orient_list = [], [], []
    pelvis_translation_list, pred_vertices_list, pred_keypoints3d_list = [], [], []
    token_entropy_list = []
    token_embeddings_list = []
    for json_file in tqdm.tqdm(json_files, desc="Loading JSONs"):
        data = load_json(json_file)
        pred_smpl_params = data["pred_smpl_params"]
        betas_list.append(np.array(pred_smpl_params["betas"])[0])
        body_pose_list.append(np.array(pred_smpl_params["body_pose"])[0])
        global_orient_list.append(np.array(pred_smpl_params["global_orient"])[0])
        pelvis_translation_list.append(np.array(data["pred_cam_t"])[0])
        pred_vertices_list.append(np.array(data["pred_vertices"])[0])
        pred_keypoints3d_list.append(np.array(data["pred_keypoints_3d"])[0])
        if "cls_logits_softmax" in data:
            softmax = np.array(data["cls_logits_softmax"])[0]
            token_entropy_list.append(np.apply_along_axis(lambda p: -np.sum(p * np.log2(p + 1e-8)), 1, softmax.T))
        token_embeddings_list.append(np.array(data["pred_smpl_params"]["token_out"])[0])
    betas_array = np.stack(betas_list)
    body_pose_array = np.stack(body_pose_list)
    global_orient_array = np.stack(global_orient_list)
    pelvis_array = np.stack(pelvis_translation_list)
    pred_vertices_array = np.stack(pred_vertices_list)
    pred_keypoints3d_array = np.stack(pred_keypoints3d_list)
    token_entropy_array = np.stack(token_entropy_list)
    token_embeddings_array = np.stack(token_embeddings_list)
    results = {}
    id_consistency = float(betas_array.std(axis=0).mean())
    pose_flat = body_pose_array.reshape(len(body_pose_array), -1)
    pose_flat_norm = pose_flat / np.linalg.norm(pose_flat, axis=1, keepdims=True)
    cos_sim = (pose_flat_norm[1:] * pose_flat_norm[:-1]).sum(axis=1)
    pose_cos = float((1.0 - cos_sim).mean())
    pose_l2 = float(np.linalg.norm(pose_flat[1:] - pose_flat[:-1], axis=1).mean())
    kp_velocity = np.linalg.norm(pred_keypoints3d_array[1:] - pred_keypoints3d_array[:-1], axis=2)
    jv_smooth = float(kp_velocity.mean())
    pose_first_diff = pose_flat[1:] - pose_flat[:-1]
    pose_second_diff = pose_first_diff[1:] - pose_first_diff[:-1]
    ang_accel = float(np.linalg.norm(pose_second_diff, axis=1).mean())
    pelvis_diff = np.linalg.norm(pelvis_array[1:] - pelvis_array[:-1], axis=1)
    global_motion = float(pelvis_diff.mean())
    pelvis_curvature = float(compute_pelvis_curvature(pelvis_array))
    mesh_chamfers = []
    for i in range(len(pred_vertices_array) - 1):
        chamfer = chamfer_distance_single(pred_vertices_array[i], pred_vertices_array[i + 1])
        mesh_chamfers.append(chamfer)
    mesh_chamfer = float(np.mean(mesh_chamfers))
    apply = lambda x: np.exp(-x) if exp_consistency else x
    results["identity_consistency"] = apply(id_consistency)
    results["pose_temporal_cosine_smoothness"] = apply(pose_cos)
    results["pose_temporal_l2_smoothness"] = apply(pose_l2)
    results["joint_velocity_smoothness"] = apply(jv_smooth)
    results["angular_acceleration_smoothness"] = apply(ang_accel)
    results["global_motion_smoothness"] = apply(global_motion)
    results["pelvis_trajectory_curvature"] = apply(pelvis_curvature)
    results["mesh_surface_chamfer_consistency"] = apply(mesh_chamfer)
    ref_betas = betas_array[:3].mean(axis=0)
    ref_pose_embed = pose_flat_norm[:3].mean(axis=0)
    ref_pelvis = pelvis_array[:3].mean(axis=0)
    ref_token_embeddings = token_embeddings_array[:3].mean(axis=0)
    identity_diffs = np.linalg.norm(betas_array - ref_betas, axis=1)
    pose_embed_diffs = np.linalg.norm(pose_flat_norm - ref_pose_embed, axis=1)
    pelvis_translation_drift = np.linalg.norm(pelvis_array - ref_pelvis, axis=1)
    token_embeddings_diffs = 1 - np.dot(token_embeddings_array, ref_token_embeddings) / (np.linalg.norm(token_embeddings_array, axis=1) * np.linalg.norm(ref_token_embeddings))
    results["ref_identity_consistency_per_frame"] = identity_diffs.tolist()
    results["ref_pose_embedding_drift_per_frame"] = pose_embed_diffs.tolist()
    results["ref_global_translation_drift_per_frame"] = pelvis_translation_drift.tolist()
    results["ref_token_embeddings_drift_per_frame"] = token_embeddings_diffs.tolist()
    betas_norm = betas_array / (np.linalg.norm(betas_array, axis=1, keepdims=True) + 1e-8)
    ref_betas_norm = ref_betas / (np.linalg.norm(ref_betas) + 1e-8)
    identity_cosine_drift = 1.0 - (betas_norm @ ref_betas_norm)
    results["ref_identity_cosine_drift_per_frame"] = identity_cosine_drift.tolist()
    kp_diff = np.linalg.norm(pred_keypoints3d_array[1:] - pred_keypoints3d_array[:-1], axis=2)
    kp_diff_mean = kp_diff.mean(axis=1)
    kp_diff_mean = np.concatenate([[0.0], kp_diff_mean])
    results["ref_joint_velocity_drift_per_frame"] = kp_diff_mean.tolist()
    results["mesh_surface_chamfer_per_frame"] = mesh_chamfers
    results["token_entropy_per_frame"] = token_entropy_array.mean(axis=1).tolist()
    results["token_entropy_avg"] = float(token_entropy_array.mean())
    results["token_embeddings_per_frame"] = token_embeddings_array.tolist()
    return results

def normalize_metrics_against_real(gen_metrics, real_metrics, per_frame_keys):
    norm_metrics = {}
    for key in per_frame_keys:
        gen_curve = np.array(gen_metrics[key])
        real_curve = np.array(real_metrics[key])
        max_real = np.max(real_curve) + 1e-8
        norm_curve = gen_curve / max_real
        norm_metrics[key + "_normalized"] = norm_curve.tolist()
    return norm_metrics

def plot_metrics(metrics, output_prefix, normalized_metrics=None):
    identity_curve = np.array(metrics["ref_identity_consistency_per_frame"])
    pose_embed_curve = np.array(metrics["ref_pose_embedding_drift_per_frame"])
    pelvis_curve = np.array(metrics["ref_global_translation_drift_per_frame"])
    token_embeddings_curve = np.array(metrics["ref_token_embeddings_drift_per_frame"])
    frames = np.arange(len(identity_curve))
    plt.figure(figsize=(15, 5))
    plt.plot(frames, identity_curve, label="Identity Consistency Drift")
    plt.plot(frames, pose_embed_curve, label="Pose Embedding Drift")
    plt.plot(frames, pelvis_curve, label="Pelvis Translation Drift")
    plt.plot(frames, token_embeddings_curve, label="Token Embeddings")
    plt.xlabel("Frame")
    plt.ylabel("Metric Value")
    plt.title("Reference Consistency/Drift Metrics Over Time")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(identity_curve.max(), pose_embed_curve.max(), pelvis_curve.max(), token_embeddings_curve.max()) * 1.1)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_curves.png", dpi=300)
    plt.close()
    if normalized_metrics is not None:
        plt.figure(figsize=(15, 5))
        plt.plot(frames, normalized_metrics["ref_identity_consistency_per_frame_normalized"], label="Identity Drift (Norm)")
        plt.plot(frames, normalized_metrics["ref_pose_embedding_drift_per_frame_normalized"], label="Pose Drift (Norm)")
        plt.plot(frames, normalized_metrics["ref_global_translation_drift_per_frame_normalized"], label="Pelvis Drift (Norm)")
        plt.plot(frames, normalized_metrics["ref_token_embeddings_drift_per_frame_normalized"], label="Token Drift (Norm)")
        plt.xlabel("Frame")
        plt.ylabel("Normalized Value")
        plt.title("Normalized Drift Metrics (Gen / Real Max) Over Time")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 2.5)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_curves_normalized.png", dpi=300)
        plt.close()

def create_drift_visualization_video(gen_video_path, real_video_path, normalized_metrics, save_path, fps):
    temp_dir = Path(save_path).parent / "temp_drift_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    drift_curves = {
        "Identity Drift (Norm)": np.array(normalized_metrics["ref_identity_consistency_per_frame_normalized"]),
        "Pose Drift (Norm)": np.array(normalized_metrics["ref_pose_embedding_drift_per_frame_normalized"]),
        "Pelvis Drift (Norm)": np.array(normalized_metrics["ref_global_translation_drift_per_frame_normalized"]),
        "Token Drift (Norm)": np.array(normalized_metrics["ref_token_embeddings_drift_per_frame_normalized"]),
    }
    n_metrics = len(drift_curves)
    num_frames = len(next(iter(drift_curves.values())))
    frames_x = np.arange(num_frames)
    plot_frame_paths = []
    for idx in range(num_frames):
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.5 * n_metrics), dpi=150)
        if n_metrics == 1:
            axes = [axes]
        for ax, (metric_name, curve) in zip(axes, drift_curves.items()):
            ax.plot(frames_x, curve, label=metric_name, color="black")
            ax.axvline(x=idx, color="red", linestyle="--", linewidth=2)
            ax.set_xlim(0, num_frames)
            ax.set_ylim(0, 2.5)
            ax.set_xlabel("Frame")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name} Over Time")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        frame_path = temp_dir / f"frame_{idx:04d}.png"
        plt.savefig(frame_path)
        plt.close(fig)
        plot_frame_paths.append(frame_path)

    gen_cap = cv2.VideoCapture(str(gen_video_path))
    real_cap = cv2.VideoCapture(str(real_video_path))
    
    # Get dimensions of both videos
    gen_width = int(gen_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gen_height = int(gen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_width = int(real_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(real_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate aspect ratios
    gen_ratio = gen_width / gen_height
    real_ratio = real_width / real_height
    
    # Determine target dimensions while maintaining aspect ratio
    if gen_ratio > real_ratio:
        # Generated video is wider, match width
        target_width = gen_width
        target_height = int(target_width / real_ratio)
    else:
        # Generated video is taller, match height
        target_height = gen_height
        target_width = int(target_height * real_ratio)
    
    # Ensure dimensions are even (required by some codecs)
    target_width = target_width + (target_width % 2)
    target_height = target_height + (target_height % 2)

    plot_frames = [imageio.imread(str(p)) for p in plot_frame_paths]
    stitched_frames = []
    idx = 0
    while True:
        ret_gen, frame_gen = gen_cap.read()
        ret_real, frame_real = real_cap.read()
        if not ret_gen or not ret_real or idx >= len(plot_frames):
            break
            
        # Convert to RGB
        frame_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)
        frame_real = cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB)
        
        # Resize real frame to match target dimensions
        frame_real = cv2.resize(frame_real, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0,1]
        frame_gen = frame_gen.astype(np.float32) / 255.0
        frame_real = frame_real.astype(np.float32) / 255.0
        
        # Process plot frame
        plot_img = plot_frames[idx].astype(np.float32) / 255.0
        if plot_img.shape[-1] == 4:
            alpha = plot_img[..., 3:4]
            rgb = plot_img[..., :3]
            plot_img = rgb * alpha + (1 - alpha)
            
        # Resize all frames to have the same width
        width = max(frame_gen.shape[1], frame_real.shape[1], plot_img.shape[1])
        plot_img = cv2.resize(plot_img, (width, plot_img.shape[0]), interpolation=cv2.INTER_AREA)
        frame_gen = cv2.resize(frame_gen, (width, frame_gen.shape[0]), interpolation=cv2.INTER_AREA)
        frame_real = cv2.resize(frame_real, (width, frame_real.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Stack frames vertically
        combined = np.vstack([plot_img, frame_gen, frame_real])
        stitched_frames.append(combined)
        idx += 1
    gen_cap.release()
    real_cap.release()
    stitched_frames = np.stack(stitched_frames)
    export_to_video(stitched_frames, save_path, fps=fps)
    shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_folder", type=str, required=True, help="Folder with generated per-frame JSONs")
    parser.add_argument("--output", type=str, required=True, help="Directory to save output metrics and plots")
    parser.add_argument("--real_json_folder", type=str, required=True, help="Folder with real video JSONs")
    parser.add_argument("--exp_consistency", action="store_true", help="Use exp(-var) style consistency metrics")
    args = parser.parse_args()
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    gen_metrics = compute_consistency_metrics(args.json_folder, exp_consistency=args.exp_consistency)
    real_metrics = compute_consistency_metrics(args.real_json_folder, exp_consistency=args.exp_consistency)
    with open(output_dir / "gen_consistency_metrics.json", "w") as f:
        json.dump(gen_metrics, f, indent=4)
    with open(output_dir / "real_consistency_metrics.json", "w") as f:
        json.dump(real_metrics, f, indent=4)
    per_frame_keys = [
        "ref_identity_consistency_per_frame",
        "ref_pose_embedding_drift_per_frame",
        "ref_global_translation_drift_per_frame",
        "ref_token_embeddings_drift_per_frame"
    ]
    normalized_metrics = normalize_metrics_against_real(gen_metrics, real_metrics, per_frame_keys)
    with open(output_dir / "gen_normalized_metrics.json", "w") as f:
        json.dump(normalized_metrics, f, indent=4)
    output_prefix = str(output_dir / "consistency")
    plot_metrics(gen_metrics, output_prefix, normalized_metrics=normalized_metrics)
    video_metadata_path = os.path.join(os.path.dirname(os.path.dirname(args.json_folder)), "metadata.json")
    video_metadata = load_json(video_metadata_path)
    fps = video_metadata["fps"] if "fps" in video_metadata else 24
    create_drift_visualization_video(
        gen_video_path=os.path.join(args.json_folder, "combined_masked_mesh.mp4"),
        real_video_path=os.path.join(args.real_json_folder, "combined_masked_mesh.mp4"),
        normalized_metrics=normalized_metrics,
        save_path=os.path.join(args.output, "drift_visualization_normalized.mp4"),
        fps=fps,
    )
    print("✅ All plots and drift video saved successfully!")

if __name__ == "__main__":
    main()