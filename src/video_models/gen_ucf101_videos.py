import argparse
import os
import torch
import json

from models.cogvideox import CogVideoXGenerator
from models.runway import RunwayGen4TurboGenerator, RunwayGen3AlphaTurboGenerator
from models.wan2_1 import Wan2_1Generator

from diffusers.utils import load_image

HACS_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101_10classes_frames"
OUTPUT_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/"


def main(args):

    if args.model == "cogvideox":
        generator = CogVideoXGenerator(
            model_name_or_path="THUDM/CogVideoX-5b-I2V", dtype=torch.bfloat16
        )
        config = {
            "prompt": "placehlder",
            "image": "placehlder",
            "num_frames": 49,
            "num_inference_steps": 50,
            "guidance_scale": 6,
            "fps": 8,
            "seed": 42,
            "output_dir": OUTPUT_DIR + "/cogvideox_videos_5",
        }
    elif args.model == "runway_gen4_turbo":
        generator = RunwayGen4TurboGenerator(model_name="gen4_turbo")
        config = {
            "model_name": "gen4_turbo",
            "duration": 5,
            "output_dir": OUTPUT_DIR + "/runway_gen4_videos_5",
            "ratio": "1280:720",
        }
    elif args.model == "runway_gen3_alpha_turbo":
        generator = RunwayGen3AlphaTurboGenerator(model_name="gen3a_turbo")
        config = {
            "model_name": "gen3a_turbo",
            "duration": 5,
            "output_dir": OUTPUT_DIR + "/runway_gen3_alpha_videos_5",
            "ratio": "1280:768",
        }
    elif args.model == "wan2_1":
        generator = Wan2_1Generator(
            model_name_or_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", dtype=torch.bfloat16
        )
        config = {
            "prompt": "placehlder",
            "image": "placehlder",
            "num_frames": 81,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
            "fps": 16,
            "seed": 42,
            "output_dir": OUTPUT_DIR + "/wan21_videos_5",
        }
    else:
        raise ValueError("Invalid model choice")

    action_folders = [os.path.join(HACS_DIR, action_folder) for action_folder in os.listdir(HACS_DIR)]
    print(f"Found {len(action_folders)} action folders in {HACS_DIR}")
    print(action_folders)

    for action_folder in action_folders:
        if not os.path.isdir(action_folder):
            print(f"Skipping {action_folder}, not a directory")
            continue
        videos = os.listdir(action_folder)

        # get 10 videos from each action folder (sorted)
        videos = sorted(os.listdir(action_folder))[:10]

        for video in videos:
            video_path = os.path.join(action_folder, video,  "video.mp4")

            # don't generate if video already exists
            temp_output_path = config["output_dir"] + "/" + action_folder.split("/")[-1] + "/" + video + "/" + "video.mp4"
            if os.path.exists(temp_output_path):
                print(f"Skipping {video} because it already exists")
                continue

            print(f"Processing {video}...")
            metadata_path = os.path.join(action_folder, video, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"Metadata file not found for {video}")
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            action = metadata.get("action", "")
            prompt = f"A person doing {action}"
            image_path = action_folder + "/" + video + "/frame_000000.jpg"

            if args.model == "cogvideox":
                image = load_image(image_path)
            else:
                image = image_path
                config["image_path"] = image_path

            config["prompt"] = prompt

            action = action_folder.split("/")[-1]
            output_dir = config["output_dir"] + "/" + action + "/" + video + "/"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Generating video for {video} with prompt: {prompt}")
            print(f"Image path: {image_path}")
            print(f"Output directory: {config['output_dir']}")

            video_config = config.copy()
            video_config["output_dir"] = output_dir
            video_config["output_root"] = output_dir

            generator.generate(image=image, config=video_config)  # FIXME: Fix the FPS!!!!!


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HACS videos")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cogvideox", "runway_gen4_turbo", "wan2_1", "runway_gen3_alpha_turbo"],
        default="cogvideox",
        help="Model to use for video generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated videos",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for video generation",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for video generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=24,
        help="Number of frames for video generation",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for generated video",
    )

    args = parser.parse_args()
    main(args)
