import os
import json
import cv2
import argparse
from typing import List

from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
from tqdm import tqdm
import numpy as np    

# at the very top of sanity_check.py (and any other entry)
import os
os.environ.setdefault("PYGLET_HEADLESS", "True")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import pyglet
pyglet.options["headless"] = True

import pyrender  # import AFTER the options above

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


def main():
    DIR = "/home/coder/projects/DATA/UCF101/ucf101"
    BGR2RGB = False

    mesh_generator = TokenHMRMeshGenerator(
        config={"side_view": True, "save_mesh": False, "full_frame": True}
    )
    print(mesh_generator)

    # ---- Determine which actions to run ----
    all_actions = sorted([d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d))])

    all_actions = ["PushUps"]

    for action in all_actions:
        action_dir = os.path.join(DIR, action)
        videos = sorted([f for f in os.listdir(action_dir) if f.lower().endswith(".mp4")])


        for video in tqdm(videos, desc=f"Processing action '{action}'", total=len(videos)):
            video_path = os.path.join(action_dir, video)
            # print(video_path)
            frames = load_all_frames(video_path, convert_bgr2rgb=BGR2RGB)
            mesh_info = mesh_generator.process_video(frames=frames)
            # print(mesh_info)

            if mesh_info:

                video_stem = os.path.splitext(video)[0]
                video_id = os.path.join(action, video_stem)  # e.g., "playing_guitar/abc123"
                meta = {
                    "action": action,
                    "video": video,
                    "source_path": video_path,
                }
                is_single = True

                print(mesh_info.keys())
                mesh_generator.generate_mesh_from_frames(frames)
                exit()

            else:
                is_single = False

if __name__ == "__main__":
    main()
