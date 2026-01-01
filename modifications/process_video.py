import os
import cv2
import numpy as np
from tqdm import tqdm
from .dwpose import DWposeDetector

VID_DIR = "data/UCF101"
POSE_SAVE_PATH = "real_kps"

def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def flatten_first_person_no_padding(bodies, hands):
    """
    Returns (120,) for first person ONLY if we have:
      - at least 18 body keypoints
      - both hands available (left+right) in some form
    Otherwise returns None (skip frame).
    """
    # ---- body must exist with >= 18 rows
    body_all = bodies.get("candidate", None)
    if body_all is None or body_all.size == 0 or body_all.shape[0] < 18:
        return None
    body = body_all[:18]  # (18,2)

    # ---- hands must exist and provide BOTH hands (no padding)
    if hands is None:
        return None
    h = np.asarray(hands)

    if h.ndim == 4:
        # (k, 2, 21, 2) -> take person 0
        if h.shape[0] < 1 or h.shape[1:] != (2, 21, 2):
            return None
        hand_pair = h[0]  # (2,21,2)

    elif h.ndim == 3:
        # (n_hands, 21, 2) -> need at least 2 hands
        if h.shape[0] < 2 or h.shape[1:] != (21, 2):
            return None
        hand_pair = np.stack([h[0], h[1]], axis=0)  # (2,21,2)

    else:
        return None

    # ---- flatten to (120,)
    return np.concatenate([body.reshape(-1), hand_pair[0].reshape(-1), hand_pair[1].reshape(-1)], axis=0)

pose = DWposeDetector()

actions = sorted(os.listdir(VID_DIR))

for action in actions:
    action_dir = os.path.join(VID_DIR, action)
    if not os.path.isdir(action_dir):
        continue

    videos = [v for v in os.listdir(action_dir) if v.endswith(".mp4") or v.endswith(".avi")]
    print(f"\nAction '{action}': {len(videos)} videos")

    for video in tqdm(videos, desc=action):
        vid_id = os.path.splitext(video)[0]
        out_file = os.path.join(POSE_SAVE_PATH, action, vid_id, "keypoints.npy")

        if os.path.exists(out_file):
            continue

        try:
            frames = load_all_frames(os.path.join(action_dir, video))
            video_kps = []

            for frame in frames:
                out, bodies, _, hands, _ = pose(frame)
                kp = flatten_first_person_no_padding(bodies, hands)
                if kp is not None:
                    video_kps.append(kp)

            # Save only frames that had complete detections (no padding)
            video_kps = np.asarray(video_kps, dtype=np.float32)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, video_kps)

        except Exception as e:
            print(f"[ERROR] {action}/{video}: {e}")