import os
import json
import cv2
import argparse
from typing import List

from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
from tqdm import tqdm
import numpy as np                         # NEW
from pathlib import Path 

BGR2RGB = False

# ---------- NEW: helpers to convert & save (lossless) ----------
def mesh_info_to_arrays(mesh_info: dict):
    """
    mesh_info: {frame_idx: {"pose":(69,), "betas":(10,), "global_orient":(3,), "vit":(D,)}}
    Returns float32 arrays preserving exact values.
    """
    frame_ids = sorted(mesh_info.keys())
    pose  = np.stack([mesh_info[i]["pose"]          for i in frame_ids]).astype(np.float32)   # [T,69]
    betas = np.stack([mesh_info[i]["betas"]         for i in frame_ids]).astype(np.float32)   # [T,10]
    gori  = np.stack([mesh_info[i]["global_orient"] for i in frame_ids]).astype(np.float32)   # [T,3]
    vit   = np.stack([mesh_info[i]["vit"]           for i in frame_ids]).astype(np.float32)   # [T,D]
    frames = np.asarray(frame_ids, dtype=np.int32)                                           # [T]
    return pose, betas, gori, vit, frames

def save_video_npz(video_id: str, mesh_info: dict, out_root="meshes_npz", meta=None) -> str:
    """
    Saves one lossless .npz per video with arrays + small JSON meta.
    Returns output path.
    """
    pose, betas, gori, vit, frames = mesh_info_to_arrays(mesh_info)
    # out_dir = Path(out_root) / Path(video_id).parent  # keep action/ subdir
    path = video_id
    # Break into components
    parts = path.split(os.sep)

    # Grab the two parts you want
    result = os.path.join(parts[-3], parts[-2])
    os.makedirs(f"{out_root}/{parts[-3]}", exist_ok=True)
    out_dir = Path(out_root) / Path(result)  # keep action/ subdir
    # out_dir.mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}.npz"
    meta_s = json.dumps(meta or {}, ensure_ascii=False)
    np.savez_compressed(
        out_path,
        pose=pose,
        betas=betas,
        global_orient=gori,
        vit=vit,
        frame_idx=frames,
        meta=meta_s
    )
    return str(out_path)
# ---------------------------------------------------------------


def load_all_frames(video_path: str, convert_bgr2rgb: bool = False) -> List:
    """Load every frame from a video into memory (no subsampling, no cap)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if convert_bgr2rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    return frames

DIR = "/home/coder/projects/video_evals/video-gen-evals/videos/runway_gen4_videos_5_only_vids"
actions = sorted(os.listdir(DIR))

mesh_generator = TokenHMRMeshGenerator(
        config={"side_view": True, "save_mesh": False, "full_frame": True}
    )
print(mesh_generator)

for action in actions:
    action_dir = os.path.join(DIR, action)
    videos = sorted([f for f in os.listdir(action_dir)])
    full_videos = [f"{os.path.join(action_dir, f)}/video.mp4" for f in videos]

    for video in tqdm(full_videos, desc=f"Processing action '{action}'", total=len(full_videos)):
        video_path = os.path.join(action_dir, video)
        frames = load_all_frames(video_path, convert_bgr2rgb=BGR2RGB)
        mesh_info = mesh_generator.process_video_test(frames=frames)
        print(f"Processed video {video_path}, got mesh_info for {len(mesh_info)} frames")
        
        if mesh_info:

            video_stem = os.path.splitext(video)[0]
            video_id = os.path.join(action, video_stem)  # e.g., "playing_guitar/abc123"
            meta = {
                "action": action,
                "video": video,
                "source_path": video_path,
            }
            out_path = save_video_npz(video_id, mesh_info, out_root="/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz_gen4", meta=meta)