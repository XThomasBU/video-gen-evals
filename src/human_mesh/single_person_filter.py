import os
import json
import cv2
import argparse
from typing import List

from .TokenHMR.mesh_generator import TokenHMRMeshGenerator


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

BASE = "results_single_k700"
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

    DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DATA/Kinetics/kinetics-dataset/k700-2020/train"
    OUTPUT_JSON_SINGLE = "kinetics700_2020_single_person_videos.json"
    OUTPUT_JSON_NOT = "kinetics700_2020_not_single_videos.json"
    OUTPUT_JSON_ERRS = "kinetics700_2020_errors.json"  # optional but handy
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

        processed = set(singles) | set(nots) | set(errs.keys())

        for video in videos:
            if video in processed:
                print(f"[SKIP] Already processed {action}/{video}")
                continue

            video_path = os.path.join(action_dir, video)
            try:
                frames = load_all_frames(video_path, convert_bgr2rgb=BGR2RGB)
                if not frames:
                    print(f"[WARN] No frames read from {video_path}; skipping.")
                    errs[video] = "no_frames"
                    save_json(err_path, errs)
                    continue

                is_single = mesh_generator.filter_single_person(frames=frames)

                if is_single:
                    singles.append(video)
                    save_json(single_path, singles)
                else:
                    nots.append(video)
                    save_json(not_path, nots)

                print(f"{video_path}: single_person={is_single} (frames={len(frames)})")

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