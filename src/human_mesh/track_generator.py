from .TokenHMR.track_generator import TokenHMRTrackGenerator
import os


tracker = TokenHMRTrackGenerator(
        checkpoint="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
        model_config="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        overrides={"render.colors": "slahmr"}
    )

DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101_10classes_frames"
OUTPUT_DIR = "videos/ucf101/tracked_real_videos"
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        video_path = os.path.join(DIR, action, video, "video.mp4")
        if os.path.exists(os.path.join(OUTPUT_DIR, action, video, f"PHALP_video_{video}.mp4")):
            print(f"Skipping {video_path} because it already exists")
            continue
        output_dir = os.path.join(OUTPUT_DIR, action, video)
        print(f"Tracking {video_path} to {output_dir}")
        tracker.run(video_path, output_dir)

DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5"
OUTPUT_DIR = "videos/ucf101/tracked_runway_gen4_videos"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        video_path = os.path.join(DIR, action, video, "video.mp4")
        output_dir = os.path.join(OUTPUT_DIR, action, video)
        if os.path.exists(os.path.join(OUTPUT_DIR, action, video, f"PHALP_video_{video}.mp4")):
            print(f"Skipping {video_path} because it already exists")
            continue
        print(f"Tracking {video_path} to {OUTPUT_DIR}")
        tracker.run(video_path, output_dir)


DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/cogvideox_videos_5"
OUTPUT_DIR = "videos/ucf101/tracked_cogvideox_videos"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        video_path = os.path.join(DIR, action, video, "video.mp4")
        output_dir = os.path.join(OUTPUT_DIR, action, video)
        if os.path.exists(os.path.join(OUTPUT_DIR, action, video, f"PHALP_video_{video}.mp4")):
            print(f"Skipping {video_path} because it already exists")
            continue
        print(f"Tracking {video_path} to {OUTPUT_DIR}")
        tracker.run(video_path, output_dir)

DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5"
OUTPUT_DIR = "videos/ucf101/tracked_wan21_videos"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        video_path = os.path.join(DIR, action, video, "video.mp4")
        output_dir = os.path.join(OUTPUT_DIR, action, video)
        if os.path.exists(os.path.join(OUTPUT_DIR, action, video, f"PHALP_video_{video}.mp4")):
            print(f"Skipping {video_path} because it already exists")
            continue
        print(f"Tracking {video_path} to {output_dir}")
        tracker.run(video_path, output_dir)