import cv2
import os
import numpy as np
import argparse
import subprocess
import shutil
import json
import re

def extract_frames(video_path, output_dir):

    print(f"Extracting frames for {video_path} to {output_dir}")
    """
    Extract all frames from a video and save them to the output directory.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory where frames will be saved
    
    Returns:
        int: Number of frames extracted
    """
    # Check if the video has already been extracted
    if os.path.exists(output_dir):
        print(f"Skipping {video_path} as it has already been extracted.")
        return 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    frame_count = 0
    
    try:
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            # Break if no frame is read
            if not ret:
                break
            
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            
    finally:
        # Release the video capture object
        cap.release()
    
    return frame_count

def get_fps(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    frame_rate_raw = result.stdout.strip()
    num, den = map(int, frame_rate_raw.split("/"))
    return num / den


def get_duration(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames from a video file')
    parser.add_argument('--video_dir', type=str, help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, help='Directory where frames will be saved')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    actions = os.listdir(args.video_dir)
    print(f"Extracting frames for {len(actions)} actions")
    actions = ["BodyWeightSquats", "HulaHoop", "JumpingJack", "PullUps", "PushUps", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "WallPushups"]
    for action in actions:
        videos = [f for f in os.listdir(os.path.join(args.video_dir, action)) if f.endswith(".mp4")]
        print(f"Extracting frames for {len(videos)} videos in {action}")
        for video in videos:
            video_path = os.path.join(args.video_dir, action, video)
            output_dir = os.path.join(args.output_dir, action, video.replace(".mp4", ""))
            num_frames = extract_frames(video_path, output_dir)
            # print(f"Extracted {num_frames} frames for {video} at {output_dir}")
            if num_frames == 0:
                print(f"Already extracted {video} at {output_dir}")
                continue

            video_id = os.path.splitext(video)[0]
            metadata_path = os.path.join(args.output_dir, action, output_dir, f"metadata.json")

            # copy video to output dir
            shutil.copy(video_path, os.path.join(args.output_dir, action, output_dir, f"video.mp4"))

            video_duration = get_duration(video_path)
            fps = get_fps(video_path)
            total_frames = num_frames
            action_name = action.replace("_", " ")
            #split between capital letters
            action_name = re.sub(r"([A-Z])", r" \1", action_name).strip()

            metadata = {
                "youtube_id": video_id,
                "fps": fps,
                "duration_seconds": video_duration,
                "total_frames": total_frames,
                "start_frame": 1,
                "end_frame": total_frames,
                "selected_frame_count": total_frames,
                "action": action_name,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            print(f"Finished {video} | Action: {action_name}")
    print("Done...All Done")

