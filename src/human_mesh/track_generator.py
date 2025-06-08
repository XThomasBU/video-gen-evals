from .TokenHMR.track_generator import TokenHMRTrackGenerator
import os


tracker = TokenHMRTrackGenerator(
        checkpoint="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
        model_config="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        overrides={"render.colors": "slahmr"}
    )

DIR = "videos/ucf101/real_videos_5"
GEN_DIR = "videos/ucf101/runway_gen4_videos_5"
OUTPUT_DIR = "videos/ucf101/tracked_real_videos_5"
actions = os.listdir(GEN_DIR)

for action in actions:
    videos = os.listdir(os.path.join(GEN_DIR, action))
    for video in videos:
        video_path = os.path.join(DIR, action, video, "video.mp4")
        output_dir = os.path.join(OUTPUT_DIR, action, video)
        tracker.run(video_path, output_dir)

DIR = "videos/ucf101/runway_gen4_videos_5"
OUTPUT_DIR = "videos/ucf101/tracked_runway_gen4_videos_5"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        video_path = os.path.join(DIR, action, video, "video.mp4")
        output_dir = os.path.join(OUTPUT_DIR, action, video)
        tracker.run(video_path, output_dir)


# DIR = "videos/ucf101/cogvideox_videos"
# OUTPUT_DIR = "videos/ucf101/tracked_cogvideox_videos"
# videos = os.listdir(DIR)
# actions = os.listdir(DIR)

# for action in actions:
#     videos = os.listdir(os.path.join(DIR, action))
#     for video in videos:
#         video_path = os.path.join(DIR, action, video, "video.mp4")
#         output_dir = os.path.join(OUTPUT_DIR, action, video)
#         tracker.run(video_path, output_dir)