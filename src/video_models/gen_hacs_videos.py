import argparse
import os
import torch
import json

from models.cogvideox import CogVideoXGenerator
from models.runway import RunwayGen4TurboGenerator

from diffusers.utils import load_image

HACS_DIR = "saved_data/hacs"

def main(args):

    if args.model == "cogvideox":
        generator = CogVideoXGenerator(model_name_or_path="THUDM/CogVideoX-5b-I2V", dtype=torch.bfloat16)
        config = {
            "prompt": 'placehlder',
            "image": 'placehlder',
            "num_frames": 49,
            "num_inference_steps": 50,
            "guidance_scale": 6,
            "fps": 8,
            "seed": 42,
            "output_dir": "saved_data/cogvideox",
        }
    elif args.model == "runway_gen4_turbo":
        generator = RunwayGen4TurboGenerator(model_name="gen4_turbo")
        config = {} # TODO: Add config for RunwayGen4TurboGenerator
    else:
        raise ValueError("Invalid model choice")

    video_ids = os.listdir(HACS_DIR)
    print(f"Found {len(video_ids)} videos in {HACS_DIR}")
    print(video_ids)

    for video_id in video_ids:
        video_dir = os.path.join(HACS_DIR, video_id)
        if not os.path.isdir(video_dir):
            print(f"Skipping {video_id}, not a directory")
            continue

        print(f"Processing {video_id}...")
        metadata_path = os.path.join(video_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found for {video_id}")
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        action = metadata.get("action", "")
        prompt = f"A person doing {action}"
        image_path = f"saved_data/hacs/{video_id}/selected_frames/frame_00001.png"

        if args.model == "cogvideox":
            image = load_image(image_path)
        else:
            image  = image_path

        config["prompt"] = prompt

        config["output_dir"] = os.path.join("saved_data", "hacs", video_id, "generated_videos")
        os.makedirs(config["output_dir"], exist_ok=True)
        print(f"Generating video for {video_id} with prompt: {prompt}")
        print(f"Image path: {image_path}")
        print(f"Output directory: {config['output_dir']}")

        generator.generate(
            image = image,
            config=config
        ) # FIXME: Fix the FPS!!!!!

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HACS videos")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cogvideox", "runway_gen4_turbo"],
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