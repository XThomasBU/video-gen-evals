import os
import json
import cv2
import argparse
from typing import List

from TokenHMR.tokenhmr.mesh_generator import TokenHMRMeshGenerator
from tqdm import tqdm
import numpy as np
from pathlib import Path 

def mesh_info_to_arrays(mesh_info: dict):
    """
    mesh_info: {frame_idx: {"pose":(69,), "betas":(10,), "global_orient":(3,), "vit":(D,)}}
    Returns float32 arrays preserving exact values.
    """
    frame_ids = sorted(mesh_info.keys())
    pose  = np.stack([mesh_info[i]["pose"]          for i in frame_ids]).astype(np.float32)
    betas = np.stack([mesh_info[i]["betas"]         for i in frame_ids]).astype(np.float32)
    gori  = np.stack([mesh_info[i]["global_orient"] for i in frame_ids]).astype(np.float32)
    vit   = np.stack([mesh_info[i]["vit"]           for i in frame_ids]).astype(np.float32)
    frames = np.asarray(frame_ids, dtype=np.int32)
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


def extract_frames_to_images(video_path: str, output_dir: str, convert_bgr2rgb: bool = False) -> List[str]:
    """Extract all frames from video and save as images. Returns list of image paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_paths = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            image_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            frame_idx += 1
    finally:
        cap.release()
    
    return image_paths

def load_frames_from_images(image_paths: List[str], convert_bgr2rgb: bool = False) -> List:
    """Load frames from saved image files."""
    frames = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        if convert_bgr2rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames

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

BASE = "UCF101_MESHES_LOGGING"
SINGLE_DIR = os.path.join(BASE, "single")
NOT_DIR    = os.path.join(BASE, "not_single")
ERR_DIR    = os.path.join(BASE, "errors")
FRAMES_DIR = os.path.join(BASE, "frames")
Path(SINGLE_DIR).mkdir(parents=True, exist_ok=True)
Path(NOT_DIR).mkdir(parents=True, exist_ok=True)
Path(ERR_DIR).mkdir(parents=True, exist_ok=True)
Path(FRAMES_DIR).mkdir(parents=True, exist_ok=True)

def load_list(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def load_dict(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            print(f"[WARN] Corrupted {path}; starting fresh.")
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    args = parse_args()

    DIR = "data/UCF101"
    OUTPUT_JSON_SINGLE = "UCF101_single_person_videos.json"
    OUTPUT_JSON_NOT = "UCF101_not_single_videos.json"
    OUTPUT_JSON_ERRS = "UCF101_errors.json"
    BGR2RGB = False

    single_person_dict = load_json(OUTPUT_JSON_SINGLE)
    not_single_dict = load_json(OUTPUT_JSON_NOT)
    errors_dict = load_json(OUTPUT_JSON_ERRS)

    mesh_generator = TokenHMRMeshGenerator(
        config={"side_view": True, "save_mesh": False, "full_frame": True}
    )
    print(mesh_generator)

    all_actions = sorted([d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d))])
    if args.action:
        if args.action not in all_actions:
            raise ValueError(f"Action '{args.action}' not found under {DIR}")
        actions = [args.action]
    else:
        actions = all_actions

    for action in actions:
        action_dir = os.path.join(DIR, action)
        videos = sorted([f for f in os.listdir(action_dir) 
                        if f.lower().endswith((".mp4", ".avi", ".mkv"))])

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
                video_stem = os.path.splitext(video)[0]
                
                frames_output_dir = os.path.join(FRAMES_DIR, action, video_stem)
                
                if os.path.exists(frames_output_dir) and len(os.listdir(frames_output_dir)) > 0:
                    image_paths = sorted([os.path.join(frames_output_dir, f) 
                                         for f in os.listdir(frames_output_dir) 
                                         if f.endswith('.jpg')])
                    frames = load_frames_from_images(image_paths, convert_bgr2rgb=BGR2RGB)
                else:
                    image_paths = extract_frames_to_images(video_path, frames_output_dir, convert_bgr2rgb=BGR2RGB)
                    frames = load_frames_from_images(image_paths, convert_bgr2rgb=BGR2RGB)
                
                mesh_info = mesh_generator.process_video(frames=frames)

                if mesh_info:
                    video_id = os.path.join(action, video_stem)
                    meta = {
                        "action": action,
                        "video": video,
                        "source_path": video_path,
                    }
                    out_path = save_video_npz(video_id, mesh_info, out_root="real_meshes", meta=meta)
                    is_single = True
                else:
                    is_single = False

                if is_single:
                    singles.append(video)
                    save_json(single_path, singles)
                else:
                    nots.append(video)
                    save_json(not_path, nots)

            except Exception as e:
                msg = str(e)
                print(f"[WARN] Failed on {video_path}: {msg}")
                errs[video] = msg
                save_json(err_path, errs)

            finally:
                del frames

    print(f"[DONE] Finished.\n  Singles: {OUTPUT_JSON_SINGLE}\n  Not singles: {OUTPUT_JSON_NOT}\n  Errors: {OUTPUT_JSON_ERRS}")


if __name__ == "__main__":
    main()
