import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------
# === CONFIG ===
CODEBOOK_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/codebook.npy"

REAL_CLASS = "ThrowDiscus"
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
GEN_ROOT_BASE = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101"

MODEL_LIST = [
    "mesh_cogvideox_videos",
    "mesh_runway_gen4_videos",
    "mesh_runway_gen3_alpha_videos",
    "mesh_opensora_256p_videos",
]

# --------------------------------------------
print("✅ Loading codebook...")
codebook = np.load(CODEBOOK_PATH).astype(np.float64)
print(f"✅ Codebook shape: {codebook.shape}")

# --------------------------------------------
# === HELPERS ===
def load_video_logits(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    all_out = []
    for p in frames:
        with open(p, "rb") as f:
            data = pickle.load(f)
        logits = np.array(data.get("cls_logits_softmax", []))
        if logits.ndim > 1:
            logits = logits[0]
        all_out.append(logits)
    return np.array(all_out) if all_out else None

def logits_to_embeddings(logits_seq, codebook):
    return np.einsum('ntc,cd->ntd', logits_seq, codebook)

def compute_velocity_magnitudes(seq_embs):
    # seq_embs: [T, D]
    v = seq_embs[1:] - seq_embs[:-1]     # [T-1, D]
    speeds = np.linalg.norm(v, axis=1)   # [T-1]
    return speeds

def compute_curvature_magnitudes(seq_embs, eps=1e-6):
    """
    Compute curvature exactly as in the paper:
    θ_i = arccos[(Δz_i · Δz_{i+1}) / (||Δz_i|| * ||Δz_{i+1}||)]
    """
    delta_z = seq_embs[1:] - seq_embs[:-1]  # [T-1, D]
    
    norms = np.linalg.norm(delta_z, axis=1)  # [T-1]
    norms = np.maximum(norms, eps)
    
    dot_products = np.sum(delta_z[:-1] * delta_z[1:], axis=1)  # [T-2]
    norm_products = norms[:-1] * norms[1:]                     # [T-2]
    cosine_similarity = dot_products / norm_products
    
    # Clip cosine similarity to [-1,1] to avoid numerical issues
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    angles = np.arccos(cosine_similarity)  # in radians
    
    # Convert to degrees (if desired, as in paper)
    angles_deg = np.degrees(angles)
    return angles_deg  # [T-2]

# --------------------------------------------
# === MAIN FLOW ===
def main():
    print(f"\n=== ACTION CLASS: {REAL_CLASS} ===")

    real_class_dir = Path(REAL_ROOT) / REAL_CLASS
    if not real_class_dir.is_dir():
        print(f"❌ Real class folder not found: {real_class_dir}")
        return

    real_velocities_all = []
    real_curvatures_all = []

    print("\n✅ Loading and processing REAL videos...")
    for real_vid in tqdm(sorted(os.listdir(real_class_dir)), desc="Real videos"):
        real_vid_path = real_class_dir / real_vid
        logits_seq = load_video_logits(real_vid_path)
        if logits_seq is None:
            continue

        # map to embeddings and flatten
        embs_seq = logits_to_embeddings(logits_seq, codebook)  # [T, D]
        embs_seq = embs_seq.mean(axis=1)
        # optionally pool / flatten any spatial dims, here it's already [T, D]
        embs_seq = embs_seq.reshape(embs_seq.shape[0], -1)  # Flatten spatial dimensions
        
        # velocities
        vel = compute_velocity_magnitudes(embs_seq)
        real_velocities_all.extend(vel.tolist())

        # curvature
        curv = compute_curvature_magnitudes(embs_seq)
        real_curvatures_all.extend(curv.tolist())

    print(f"✅ Mean REAL speed    : {np.mean(real_velocities_all):.4f}")
    print(f"✅ Mean REAL curvature: {np.mean(real_curvatures_all):.4f}")

    print("\nProcessing GENERATED videos...")
    for model in MODEL_LIST:
        gen_class_dir = Path(GEN_ROOT_BASE) / model / REAL_CLASS
        if not gen_class_dir.is_dir():
            print(f"❌ Generated class folder not found: {gen_class_dir}")
            continue

        print(f"\n=== MODEL: {model} ===")
        gen_velocities_all = []
        gen_curvatures_all = []

        print("\n✅ Loading and processing GENERATED videos...")
        for gen_vid in tqdm(sorted(os.listdir(gen_class_dir)), desc="Generated videos"):
            gen_vid_path = gen_class_dir / gen_vid
            logits_seq = load_video_logits(gen_vid_path)
            if logits_seq is None:
                continue
            
            embs_seq = logits_to_embeddings(logits_seq, codebook)
            # embs_seq = embs_seq.mean(axis=1)
            embs_seq = embs_seq.reshape(embs_seq.shape[0], -1)  # Flatten spatial dimensions

            # velocities
            vel = compute_velocity_magnitudes(embs_seq)
            gen_velocities_all.extend(vel.tolist())

            # curvature
            curv = compute_curvature_magnitudes(embs_seq)
            gen_curvatures_all.extend(curv.tolist())

        print(f"✅ Mean {model} speed    : {np.mean(gen_velocities_all):.4f}")
        print(f"✅ Mean {model} curvature: {np.mean(gen_curvatures_all):.4f}")

if __name__ == "__main__":
    main()