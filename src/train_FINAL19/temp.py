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
from save_data import  extract_windows
from train import (
    # TemporalTransformer,
    # load_video_sequence,
    # extract_windows,
    # collate_fn,
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)
from models import TemporalTransformerV2Plus
from utils import *
from tabulate import tabulate

ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
# ALL_CLASSES += ["BoxingPunchingBag", "CleanAndJerk", "GolfSwing", "HandstandPushups", "JugglingBalls", "Lunges", "WritingOnBoard"]



INPUT_DIM= 1370


# ---------- core utilities ----------
def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def zscore_time(X: torch.Tensor) -> torch.Tensor:
    """
    Z-score per feature across time (dim=0).
    X: [T, D]
    """
    mean = X.mean(dim=0, keepdim=True)
    std  = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (X - mean) / std

def hann_window(T: int, device=None) -> torch.Tensor:
    # PyTorch Hann window without FFT dependencies
    n = torch.arange(T, device=device, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * n / (T - 1))

def rfft_psd(X: torch.Tensor, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided power spectral density per feature by rFFT along time.
    X: [T, D] (already normalized & windowed)
    Returns:
      freqs: [F] in Hz
      psd:   [F, D] power for each feature
    """
    T = X.shape[0]
    # real FFT along time dim (0). Output shape [F, D]
    Xf = torch.fft.rfft(X, dim=0)  # complex
    power = (Xf.real ** 2 + Xf.imag ** 2) / T  # simple periodogram
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps)
    return _to_numpy(freqs), _to_numpy(power)

def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    psd: [F, D] or [F] -> we aggregate across features if needed.
    Returns scalar centroid in Hz.
    """
    if psd.ndim == 2:
        psd_agg = psd.mean(axis=1)
    else:
        psd_agg = psd
    num = (freqs * psd_agg).sum()
    den = psd_agg.sum() + 1e-12
    return float(num / den)

def high_freq_ratio(freqs: np.ndarray, psd: np.ndarray, cutoff_hz: float) -> float:
    """
    Fraction of power above cutoff_hz (averaged over features if needed).
    """
    if psd.ndim == 2:
        psd_agg = psd.mean(axis=1)
    else:
        psd_agg = psd
    mask_hi = freqs >= cutoff_hz
    total = psd_agg.sum() + 1e-12
    hi = psd_agg[mask_hi].sum()
    return float(hi / total)

# ---------- main analysis ----------
def analyze_modalities(mods: dict[str, torch.Tensor],
                       fps: float = 30.0,
                       cutoff_hz: float | None = None,
                       show_plots: bool = True) -> dict:
    """
    mods: dict of {name: tensor[T, D]} e.g. {"vit": vit, "go": go, "pose": pose, "betas": betas, "kp2d": kp2d}
    fps: frames per second for your video
    cutoff_hz: if None, set to 0.25 * Nyquist (a conservative 'high freq' threshold)
    Returns a dict of metrics per modality.
    """
    results = {}
    for name, X in mods.items():
        assert X.ndim == 2 and X.shape[0] >= 2, f"{name}: expected [T, D] with T>=2"
        T = X.shape[0]
        device = X.device

        # 1) normalize across time (per feature), 2) soften edges with Hann window to reduce leakage
        Xn = zscore_time(X)
        w = hann_window(T, device=device).view(T, 1)
        Xw = Xn * w

        # FFT-based PSD
        freqs, psd = rfft_psd(Xw, fps=fps)  # freqs: [F], psd: [F, D]
        nyquist = freqs.max()
        th = cutoff_hz if cutoff_hz is not None else 0.25 * nyquist

        # Metrics
        centroid = spectral_centroid(freqs, psd)
        hfr = high_freq_ratio(freqs, psd, cutoff_hz=th)

        # Aggregate spectra (mean & median over features)
        psd_mean = psd.mean(axis=1)
        psd_median = np.median(psd, axis=1)

        results[name] = {
            "freqs_hz": freqs,                 # [F]
            "psd_mean": psd_mean,              # [F]
            "psd_median": psd_median,          # [F]
            "spectral_centroid_hz": centroid,  # scalar
            "high_freq_ratio": hfr,            # scalar
            "cutoff_hz": th,
            "nyquist_hz": nyquist,
            "T": T,
        }

        # if show_plots:
        #     # PSD (mean over features)
        #     plt.figure(figsize=(7, 4))
        #     plt.plot(freqs, psd_mean)
        #     plt.title(f"{name} – mean PSD over features (T={T}, fps={fps:.2f})")
        #     plt.xlabel("Frequency (Hz)")
        #     plt.ylabel("Power")
        #     plt.axvline(th, linestyle="--")
        #     plt.tight_layout()
        #     plt.show()

        #     # Optional: median curve too (can reveal robustness to outliers)
        #     plt.figure(figsize=(7, 4))
        #     plt.plot(freqs, psd_median)
        #     plt.title(f"{name} – median PSD over features")
        #     plt.xlabel("Frequency (Hz)")
        #     plt.ylabel("Power")
        #     plt.axvline(th, linestyle="--")
        #     plt.tight_layout()
        #     plt.show()

    return results


def plot_psd(results, name="kp2d", model="model"):
    """
    Plot mean vs median PSD for a given modality.
    results: dict from analyze_modalities (with freqs_hz, psd_mean, psd_median)
    name: key of the modality to plot
    """
    res = results[name]
    freqs = res["freqs_hz"]
    psd_mean = res["psd_mean"]
    psd_median = res["psd_median"]

    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd_mean, label="Mean PSD", color="blue")
    plt.plot(freqs, psd_median, label="Median PSD", color="orange")
    plt.axvline(res["cutoff_hz"], linestyle="--", color="red", label=f"Cutoff @ {res['cutoff_hz']:.1f} Hz")
    plt.title(f"{name} Power Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"PSD_{name}_{model}.png")

def load_video_sequence(video_folder, MESH_DIR, POSE_DIR, stats_dir, FPS):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace(MESH_DIR, POSE_DIR)
    twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

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

    # [T,dim] tensors (RAW, unnormalized)
    vit   = torch.stack(vit_list, dim=0)
    go    = torch.stack(go_list, dim=0)
    pose  = torch.stack(pose_list, dim=0)
    betas = torch.stack(betas_list, dim=0)
    kp2d  = torch.stack(kp_list, dim=0)

    print(vit.shape, go.shape, pose.shape, betas.shape, kp2d.shape)
    results = analyze_modalities({
        "vit": vit,
        "go": go,
        "pose": pose,
        "betas": betas,
        "kp2d": kp2d,
    }, fps=FPS, cutoff_hz=2.0, show_plots=True)
    return results


VID_PATH = {
    'wan21': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5',
    'runway_gen3_alpha': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen3_alpha_videos_5',
    'runway_gen4': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5',
    'hunyuan_360p': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/hunyuan_videos_360p_formatted',
    'opensora_256p':  '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/opensora_videos_256p_formatted',
    'cogvideox':  '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/cogvideox_videos_5'
}

# for MODEL in ["wan21", "runway_gen4", "hunyuan_360p", "opensora_256p", "cogvideox"]:
for MODEL in ["hunyuan_360p"]:
    GEN_ROOT = f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_{MODEL}_videos"
    if "cogvideox" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        FPS = 8.0
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_COGVIDEOX"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos"
    elif "opensora" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        FPS = 24.0
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_OPENSORA_256p"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_opensora_256p_videos"
    elif "hunyuan" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        FPS = 24.0
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_HUNYUAN_360p"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_hunyuan_360p_videos"
    elif "runway" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        FPS = 24.0
        if MODEL == "runway_gen3_alpha":
            POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN3_ALPHA"
            MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen3_alpha_videos"
        elif MODEL == "runway_gen4":
            POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_RUNWAY_GEN4"
            MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos"
    elif "wan21" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
        FPS = 16
        POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_WAN21"
        MESH_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_wan21_videos"
    else:
        WINDOW_SIZE = 32
        STRIDE = 8
    BATCH_SIZE = 64

    # ————— load train embeddings & centroids —————
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

    # ————— load model —————
    print(" Loading model...")
    model = TemporalTransformerV2Plus(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)
    print(f" Loading model from temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_NO_ENT.pt")
    model.load_state_dict(torch.load(f"SAVE/temporal_transformer_model_window_32_stride_8_valid_window_NO_ENT.pt"))
    model.eval()

    for cls in ALL_CLASSES:
        root = GEN_ROOT
        for vid in os.listdir(Path(root) / cls):
            results = load_video_sequence(Path(root) / cls / vid, MESH_DIR, POSE_DIR, "SAVE", FPS)
            print(results)
            plot_psd(results, name="kp2d", model=MODEL)
            exit()