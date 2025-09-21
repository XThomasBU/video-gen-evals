import os
import json
import cv2
import argparse
from typing import List

from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
from tqdm import tqdm
import numpy as np                         # NEW
from pathlib import Path 

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
    out_dir = Path(out_root) / Path(video_id).parent  # keep action/ subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video_id).name}.npz"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Filter single-person videos")
    parser.add_argument(
        "--action",
        type=str,
        default=None,
        help="Optional: only process this action (directory name). If not set, process all actions."
    )
    return parser.parse_args()


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

import os, json, pathlib

BASE = "FINAL_MESH_K700_2020"
SINGLE_DIR = os.path.join(BASE, "single")
NOT_DIR    = os.path.join(BASE, "not_single")
ERR_DIR    = os.path.join(BASE, "errors")
pathlib.Path(SINGLE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(NOT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ERR_DIR).mkdir(parents=True, exist_ok=True)

def load_list(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []  # lists per action

def load_dict(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            print(f"[WARN] Corrupted {path}; starting fresh.")
    return {}  # dict of errors per action

def save_json(path, data):
    # simple, per-file write (small files, one writer per action)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    args = parse_args()

    DIR = "/home/coder/projects/DATA/kinetics-dataset/k700-2020/train"
    OUTPUT_JSON_SINGLE = "K700_single_person_videos.json"
    OUTPUT_JSON_NOT = "K700_not_single_videos.json"
    OUTPUT_JSON_ERRS = "K700_errors.json"  # optional but handy
    BGR2RGB = False

    # ---- Load existing progress if available ----
    single_person_dict = load_json(OUTPUT_JSON_SINGLE)
    not_single_dict = load_json(OUTPUT_JSON_NOT)
    errors_dict = load_json(OUTPUT_JSON_ERRS)

    mesh_generator = TokenHMRMeshGenerator(
        config={"side_view": True, "save_mesh": False, "full_frame": True}
    )
    print(mesh_generator)

    # ---- Determine which actions to run ----
    all_actions = sorted([d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d))])
    if args.action:
        if args.action not in all_actions:
            raise ValueError(f"Action '{args.action}' not found under {DIR}")
        actions = [args.action]
    else:
        actions = all_actions

    for action in actions:
        action_dir = os.path.join(DIR, action)
        videos = sorted([f for f in os.listdir(action_dir) if f.lower().endswith(".mp4")])

        # per-action files
        single_path = os.path.join(SINGLE_DIR, f"{action}.json")
        not_path    = os.path.join(NOT_DIR,    f"{action}.json")
        err_path    = os.path.join(ERR_DIR,    f"{action}.json")

        singles = load_list(single_path)
        nots    = load_list(not_path)
        errs    = load_dict(err_path)

        processed = set(singles) | set(nots) 

        for video in tqdm(videos, desc=f"Processing action '{action}'", total=len(videos)):
            if video in processed:
                print(f"[SKIP] {action}/{video} (already processed)")
                continue

            try:
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
                    out_path = save_video_npz(video_id, mesh_info, out_root="/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz", meta=meta)

                    is_single = True

                else:
                    is_single = False

                # video_path = os.path.join(action_dir, video)
                # try:
                #     frames = load_all_frames(video_path, convert_bgr2rgb=BGR2RGB)
                #     if not frames:
                #         print(f"[WARN] No frames read from {video_path}; skipping.")
                #         errs[video] = "no_frames"
                #         save_json(err_path, errs)
                #         continue

                if is_single:
                    singles.append(video)
                    save_json(single_path, singles)
                else:
                    nots.append(video)
                    save_json(not_path, nots)

                #     print(f"{video_path}: single_person={is_single} (frames={len(frames)})")

            except Exception as e:
                msg = str(e)
                print(f"[WARN] Failed on {video_path}: {msg}")
                errs[video] = msg
                save_json(err_path, errs)

            finally:
                del frames  # free memory

    print(f"[DONE] Finished.\n  Singles: {OUTPUT_JSON_SINGLE}\n  Not singles: {OUTPUT_JSON_NOT}\n  Errors: {OUTPUT_JSON_ERRS}")


if __name__ == "__main__":
    main()
