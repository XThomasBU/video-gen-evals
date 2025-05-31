from .TokenHMR.track_generator import TokenHMRTrackGenerator
import os


tracker = TokenHMRTrackGenerator(
        checkpoint="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
        model_config="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        overrides={"render.colors": "slahmr"}
    )

DIR = "saved_data/ucf101"
videos = os.listdir(DIR)

# for video in videos:
#     video_path = os.path.join(DIR, video, f"{video}_full.mp4")
#     output_dir = os.path.join(DIR, video, "real_video_tracking_output")
#     tracker.run(video_path, output_dir)

# for video in videos:
#     video_path = os.path.join(DIR, video, "generated_videos_cogvideox", f"video.mp4")
#     output_dir = os.path.join(DIR, video, "cogvideox_tracking_output")
#     tracker.run(video_path, output_dir)
#     # break

# for video in videos:
#     video_path = os.path.join(DIR, video, "generated_videos_runway_gen4_turbo", f"video.mp4")
#     output_dir = os.path.join(DIR, video, "runway_gen4_turbo_tracking_output")
#     tracker.run(video_path, output_dir)
#     # break

video_path = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/generated_videos_cogvideox/cog_hula.mp4"
output_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/cogvideox_tracking_output_threshold50"
tracker.run(video_path, output_dir)