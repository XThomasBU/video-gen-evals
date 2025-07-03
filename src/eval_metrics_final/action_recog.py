import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
from scipy.linalg import sqrtm

# ——— CONFIG ———
CODEBOOK_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/codebook.npy"
N_CLUSTERS    = 512
PCA_COMPONENTS = 10
TEMPERATURE   = 0.5

print("Loading codebook...")
codebook = np.load(CODEBOOK_PATH).astype(np.float64)
print(f"Codebook shape: {codebook.shape}")

# ——— DATA LOADING ———
def load_video_logits(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    all_out = []
    for p in tqdm(frames, desc=f"Loading frames in {video_folder.name}"):
        with open(p, "rb") as f:
            data = pickle.load(f)
        logits = np.array(data.get("cls_logits_softmax", []))
        if logits.ndim > 1:
            logits = logits[0]
        all_out.append(logits)
    return all_out

# ——— LOGITS → EMBEDDINGS ———
def logits_to_embeddings(logits_seq, codebook):
    out = []
    for logits in logits_seq:
        if logits.size == 0:
            continue
        emb_frame = logits @ codebook
        out.append(emb_frame)
    return np.array(out, dtype=np.float64) if len(out) > 0 else None

# ——— CODEBOOK CLUSTERING ———
def cluster_codebook_vectors(codebook, n_clusters=N_CLUSTERS):
    print("\n=== Clustering codebook ===")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(codebook)
    return kmeans

def map_to_cluster_ids(emb_seq, kmeans_model):
    num_frames, num_tokens, D = emb_seq.shape
    flattened = emb_seq.reshape(-1, D).astype(np.float64)
    labels = kmeans_model.predict(flattened)
    return labels.reshape(num_frames, num_tokens)

# ——— TRANSITION MATRIX ———
def compute_transition_matrix(cluster_ids_seq, n_clusters):
    matrix = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    for f1, f2 in zip(cluster_ids_seq[:-1], cluster_ids_seq[1:]):
        for t1, t2 in zip(f1, f2):
            matrix[t1, t2] += 1
    matrix /= (matrix.sum() + 1e-8)
    return matrix.flatten()

# ——— PCA TRAJECTORY ———
def build_pca_on_class(embeddings_list, n_components=PCA_COMPONENTS):
    print("\n=== Fitting PCA on Real Videos ===")
    all_frames = np.vstack(embeddings_list)
    scaler = StandardScaler()
    all_frames = scaler.fit_transform(all_frames)
    pca = PCA(n_components=n_components)
    pca.fit(all_frames)
    return scaler, pca

def get_video_pca_trajectory(embeddings, scaler, pca):
    num_frames = embeddings.shape[0]
    flat_embs = embeddings.reshape(num_frames, -1)
    flat_embs = scaler.transform(flat_embs)
    traj = pca.transform(flat_embs)
    return traj

# ——— TRAJECTORY METRICS ———
def compute_frechet_distance(traj1, traj2):
    mu1, cov1 = traj1.mean(0), np.cov(traj1.T)
    mu2, cov2 = traj2.mean(0), np.cov(traj2.T)
    mean_diff = np.sum((mu1 - mu2) ** 2)
    sqrt_cov = sqrtm(cov1 @ cov2 + 1e-8 * np.eye(cov1.shape[0]))
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real
    cov_dist = np.trace(cov1 + cov2 - 2 * sqrt_cov)
    return mean_diff + cov_dist

def compute_variance_energy(traj):
    return np.trace(np.cov(traj.T))

def compute_jerk(traj):
    if len(traj) < 4:
        return 0.0
    dt1 = np.diff(traj, axis=0)
    dt2 = np.diff(dt1, axis=0)
    dt3 = np.diff(dt2, axis=0)
    return np.mean(np.linalg.norm(dt3, axis=1))

# ——— MAIN PIPELINE ———
def evaluate_generated_video(gen_video_folder, codebook, kmeans_codebook, scaler, pca_model,
                             real_mean_transition, real_transitions, real_transition_mean, real_transition_std,
                             real_pca_trajectories, real_frechet_mean, real_frechet_std,
                             real_variance_mean, real_variance_std,
                             real_jerk_mean, real_jerk_std):
    logits_seq = load_video_logits(gen_video_folder)
    gen_embs_seq = logits_to_embeddings(logits_seq, codebook)
    if gen_embs_seq is None or len(gen_embs_seq) < 2:
        print(f"Invalid or too short generated video: {gen_video_folder.name}")
        return None

    # Transition matrix
    gen_cluster_ids_seq = map_to_cluster_ids(gen_embs_seq, kmeans_codebook)
    gen_transition = compute_transition_matrix(gen_cluster_ids_seq, N_CLUSTERS)
    transition_jsd = jensenshannon(real_mean_transition, gen_transition, base=2)
    transition_jsd_z = (transition_jsd - real_transition_mean) / (real_transition_std + 1e-8)

    # Trajectory
    gen_flat_frames = gen_embs_seq.reshape(len(gen_embs_seq), -1)
    gen_pca_traj = get_video_pca_trajectory(gen_flat_frames, scaler, pca_model)

    # Frechet distances
    frechet_dists = [compute_frechet_distance(gen_pca_traj, real_traj) for real_traj in real_pca_trajectories]
    mean_frechet = np.mean(frechet_dists)
    frechet_z = (mean_frechet - real_frechet_mean) / (real_frechet_std + 1e-8)

    # Variance & jerk
    gen_variance = compute_variance_energy(gen_pca_traj)
    gen_variance_z = (gen_variance - real_variance_mean) / (real_variance_std + 1e-8)

    jerk_gen = compute_jerk(gen_pca_traj)
    jerk_z = (jerk_gen - real_jerk_mean) / (real_jerk_std + 1e-8)

    return {
        'transition_jsd': transition_jsd,
        'transition_jsd_z': transition_jsd_z,
        'frechet': mean_frechet,
        'frechet_z': frechet_z,
        'variance': gen_variance,
        'variance_z': gen_variance_z,
        'jerk': jerk_gen,
        'jerk_z': jerk_z
    }

def main(real_root, gen_video_folder_a, gen_video_folder_b):
    gen_class = Path(gen_video_folder_a).parent.name
    print(f"\n=== ACTION CLASS: {gen_class} ===")

    # --- Codebook Clustering
    kmeans_codebook = cluster_codebook_vectors(codebook, n_clusters=N_CLUSTERS)

    # --- Load REAL videos
    real_class_dir = Path(real_root) / gen_class
    real_embs_list, real_transitions, real_flat_frames = [], [], []

    print("\n=== Loading REAL videos ===")
    for vid in sorted(os.listdir(real_class_dir)):
        vid_path = real_class_dir / vid
        logits_seq = load_video_logits(vid_path)
        if logits_seq and len(logits_seq) > 1:
            embs_seq = logits_to_embeddings(logits_seq, codebook)
            if embs_seq is None:
                continue
            real_embs_list.append(embs_seq)
            cluster_ids_seq = map_to_cluster_ids(embs_seq, kmeans_codebook)
            transition = compute_transition_matrix(cluster_ids_seq, N_CLUSTERS)
            real_transitions.append(transition)
            flat_embs = embs_seq.reshape(len(embs_seq), -1)
            real_flat_frames.append(flat_embs)

    if len(real_embs_list) < 2:
        print("Not enough real videos for reference!")
        return

    real_mean_transition = np.mean(real_transitions, axis=0)

    # Compute stats for transition JSD
    real_jsds = []
    for i in range(len(real_transitions)):
        for j in range(i+1, len(real_transitions)):
            jsd = jensenshannon(real_transitions[i], real_transitions[j], base=2)
            real_jsds.append(jsd)
    real_transition_mean = np.mean(real_jsds)
    real_transition_std = np.std(real_jsds)

    # --- PCA Model
    scaler, pca_model = build_pca_on_class(real_flat_frames)
    real_pca_trajectories = [get_video_pca_trajectory(f, scaler, pca_model) for f in real_flat_frames]

    # Frechet stats
    real_frechet_dists = []
    for i in range(len(real_pca_trajectories)):
        for j in range(i+1, len(real_pca_trajectories)):
            d = compute_frechet_distance(real_pca_trajectories[i], real_pca_trajectories[j])
            real_frechet_dists.append(d)
    real_frechet_mean = np.mean(real_frechet_dists)
    real_frechet_std = np.std(real_frechet_dists)

    # Variance & jerk stats
    real_variances = [compute_variance_energy(r) for r in real_pca_trajectories]
    real_jerks = [compute_jerk(r) for r in real_pca_trajectories]

    mean_real_variance = np.mean(real_variances)
    std_real_variance = np.std(real_variances)

    mean_real_jerk = np.mean(real_jerks)
    std_real_jerk = np.std(real_jerks)

    # --- Evaluate Gen Videos
    print("\n=== Evaluating Generated Video A ===")
    results_a = evaluate_generated_video(
        Path(gen_video_folder_a), codebook, kmeans_codebook, scaler, pca_model,
        real_mean_transition, real_transitions, real_transition_mean, real_transition_std,
        real_pca_trajectories, real_frechet_mean, real_frechet_std,
        mean_real_variance, std_real_variance,
        mean_real_jerk, std_real_jerk
    )

    print("\n=== Evaluating Generated Video B ===")
    results_b = evaluate_generated_video(
        Path(gen_video_folder_b), codebook, kmeans_codebook, scaler, pca_model,
        real_mean_transition, real_transitions, real_transition_mean, real_transition_std,
        real_pca_trajectories, real_frechet_mean, real_frechet_std,
        mean_real_variance, std_real_variance,
        mean_real_jerk, std_real_jerk
    )

    # --- Report
    print("\n=== MOTION SIGNATURE COMPARISON ===")
    for label, res in [('A', results_a), ('B', results_b)]:
        if res is None:
            continue
        print(f"\n--- Video {label} ---")
        print(f"Transition JSD: {res['transition_jsd']:.4f} (Z-score: {res['transition_jsd_z']:.2f})")
        print(f"Frechet Distance: {res['frechet']:.4f} (Z-score: {res['frechet_z']:.2f})")
        print(f"Variance: {res['variance']:.4f} (Z-score: {res['variance_z']:.2f})")
        print(f"Jerk: {res['jerk']:.4f} (Z-score: {res['jerk_z']:.2f})")

# ——— ENTRY POINT ———
if __name__ == "__main__":
    REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
    GEN_VIDEO_FOLDER_A = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen3_alpha_videos/PullUps/v_PullUps_g17_c03"
    GEN_VIDEO_FOLDER_B = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/PullUps/v_PullUps_g17_c03"
    main(REAL_ROOT, GEN_VIDEO_FOLDER_A, GEN_VIDEO_FOLDER_B)