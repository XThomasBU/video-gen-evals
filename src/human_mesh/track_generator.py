from .TokenHMR.track_generator import TokenHMRTrackGenerator
import os


tracker = TokenHMRTrackGenerator(
        checkpoint="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
        model_config="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        overrides={"render.colors": "slahmr"}
    )

# Run tracking on a video
tracker.run("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_JumpingJack_g20_c01/v_JumpingJack_g20_c01_full.mp4")