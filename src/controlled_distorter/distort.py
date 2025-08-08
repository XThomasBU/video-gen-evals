import numpy as np
import cv2
import os
import subprocess
import imageio

# ——— Distortion Functions ———

def reverse_video(video):
    """Fully reverse a video."""
    return video[::-1]

def shuffle_video(video, scale=1.0):
    """Shuffle a fraction of the video frames."""
    if not (0.0 <= scale <= 1.0):
        raise ValueError("Scale must be between 0.0 and 1.0")

    video = video.copy()
    n_frames = len(video)
    if scale == 0.0:
        return video

    n_to_shuffle = int(n_frames * scale)
    shuffle_indices = np.random.choice(n_frames, n_to_shuffle, replace=False)
    selected = [video[i] for i in shuffle_indices]
    np.random.shuffle(selected)
    for idx, new_frame in zip(shuffle_indices, selected):
        video[idx] = new_frame
    return video

def repeat_frames(video, n_repeats=2):
    """
    Repeat each frame in the video a specified number of times.

    Args:
        video (list): List of video frames.
        n_repeats (int): Number of times to repeat each frame.

    Returns:
        list: Video frames with each frame repeated.
    """
    return [frame for frame in video for _ in range(n_repeats)]

def shuffle_window_order_scaled(video, window_size=10, scale=1.0):
    """
    Shuffle a fraction of non-overlapping windows of frames, based on the given scale.

    Args:
        video (list): List of video frames.
        window_size (int): Number of frames per window.
        scale (float): Fraction (0.0–1.0) of windows to randomly reorder.

    Returns:
        list: Video frames with selected windows shuffled.
    """
    video = video.copy()
    n_frames = len(video)

    # Step 1: Chunk into windows
    windows = [video[i:i + window_size] for i in range(0, n_frames, window_size)]
    num_windows = len(windows)

    if num_windows <= 1 or scale == 0.0:
        return video  # nothing to shuffle

    # Step 2: Choose subset of windows to shuffle
    n_shuffle = int(num_windows * scale)
    shuffle_indices = np.random.choice(num_windows, n_shuffle, replace=False)

    # Step 3: Extract & shuffle those windows
    to_shuffle = [windows[i] for i in shuffle_indices]
    np.random.shuffle(to_shuffle)

    # Step 4: Place shuffled windows back into original positions
    for i, idx in enumerate(shuffle_indices):
        windows[idx] = to_shuffle[i]

    # Step 5: Flatten back into frames
    shuffled_video = [frame for window in windows for frame in window]
    return shuffled_video

def drop_frames_fixed(video, interval=2):
    """Keep every `interval`-th frame."""
    return [f for i, f in enumerate(video) if i % interval == 0]

# ——— Frame Extraction ———

def extract_frames(video_path):
    """Extract frames from a video and convert to RGB."""
    frame_list = []
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame_list.append(frame_rgb)
    finally:
        cap.release()
    return frame_list

# ——— FPS Extraction ———

def get_fps(video_path):
    """Get the frame rate (fps) of the input video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    num, den = map(int, result.stdout.strip().split("/"))
    return num / den

# ——— Video Writer ———

def save_video(frames, path, fps):
    """Save a list of RGB frames to a video using imageio."""
    frames = [np.ascontiguousarray(f.astype(np.uint8)) for f in frames if isinstance(f, np.ndarray)]
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

# ——— Paths ———

ORG_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101_10classes_frames/Shotput/v_Shotput_g06_c01/video.mp4"
SAVE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/DEL"
os.makedirs(SAVE_DIR, exist_ok=True)

# ——— Load + Process ———

FPS = get_fps(ORG_PATH)
ORG_FRAMES = extract_frames(ORG_PATH)
print(f" Extracted {len(ORG_FRAMES)} frames at {FPS:.2f} FPS")

# ——— Export Variants ———

# Original
save_video(ORG_FRAMES, os.path.join(SAVE_DIR, "original_video.mp4"), fps=FPS)

# Reversed
REVERSED_FRAMES = reverse_video(ORG_FRAMES)
save_video(REVERSED_FRAMES, os.path.join(SAVE_DIR, "reversed_video.mp4"), fps=FPS)

# Shuffled (various scales)
for scale in [0.1, 0.3, 0.5, 0.7, 0.9]:
    shuffled_frames = shuffle_video(ORG_FRAMES, scale=scale)
    save_path = os.path.join(SAVE_DIR, f"shuffled_{int(scale*100):02d}_video.mp4")
    save_video(shuffled_frames, save_path, fps=FPS)

# Shuffled Window Order (various scales)
for scale in [0.1, 0.3, 0.5, 0.7, 0.9]:
    windowed_frames = shuffle_window_order_scaled(ORG_FRAMES, window_size=10, scale=scale)
    save_path = os.path.join(SAVE_DIR, f"windowed_shuffled_{int(scale*100):02d}_video.mp4")
    save_video(windowed_frames, save_path, fps=FPS)

for interval in [2, 3, 4, 5]:
    dropped_frames = drop_frames_fixed(ORG_FRAMES, interval=interval)
    save_path = os.path.join(SAVE_DIR, f"dropped_{interval}_frames_video.mp4")
    save_video(dropped_frames, save_path, fps=FPS)

# Repeated Frames
REPEATED_FRAMES = repeat_frames(ORG_FRAMES, n_repeats=2)
save_video(REPEATED_FRAMES, os.path.join(SAVE_DIR, "repeated_frames_2_video.mp4"), fps=FPS)

REPEATED_FRAMES = repeat_frames(ORG_FRAMES, n_repeats=3)
save_video(REPEATED_FRAMES, os.path.join(SAVE_DIR, "repeated_frames_3_video.mp4"), fps=FPS)


REPEATED_FRAMES = repeat_frames(ORG_FRAMES, n_repeats=4)
save_video(REPEATED_FRAMES, os.path.join(SAVE_DIR, "repeated_frames_4_video.mp4"), fps=FPS)


REPEATED_FRAMES = repeat_frames(ORG_FRAMES, n_repeats=5)
save_video(REPEATED_FRAMES, os.path.join(SAVE_DIR, "repeated_frames_5_video.mp4"), fps=FPS)

print(" All distorted videos saved with correct colors.")