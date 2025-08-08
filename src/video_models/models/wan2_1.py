import os
import json
import numpy as np
import torch
from datetime import datetime
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from .base import BaseHFVideoGenerator
from PIL import Image

class Wan2_1Generator(BaseHFVideoGenerator):
    def load_model(self):
        # Load VAE and image encoder
        self.image_encoder = CLIPVisionModel.from_pretrained(
            self.model_name_or_path, subfolder="image_encoder", torch_dtype=torch.float32
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_name_or_path, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_name_or_path,
            vae=self.vae,
            image_encoder=self.image_encoder,
            torch_dtype=self.dtype,
        )
        self.pipe.to("cuda")

    def generate(self, image, config: dict):
        if self.pipe is None:
            self.load_model()
            self.optimize_pipeline()

        prompt = config["prompt"]
        negative_prompt = config.get("negative_prompt", "")
        num_frames = config.get("num_frames", 81)
        guidance_scale = config.get("guidance_scale", 5.0)
        fps = config.get("fps", 16)
        output_dir = config.get("output_dir", None)

        image = load_image(image)

        # Compute image dimensions
        max_area = 480 * 832
        aspect_ratio = int(image.height) / int(image.width)
        mod_value = (
            self.pipe.vae_scale_factor_spatial
            * self.pipe.transformer.config.patch_size[1]
        )
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        )

        video_frames = result.frames[0]

        run_dir = output_dir or self._create_unique_output_dir()
        os.makedirs(run_dir, exist_ok=True)

        for i, frame in enumerate(video_frames):
            # frame.save(os.path.join(run_dir, f"frame_{i:06d}.png"))
            Image.fromarray(frame).save(os.path.join(run_dir, f"frame_{i:06d}.png"))

        video_path = os.path.join(run_dir, "video.mp4")
        export_to_video(video_frames, video_path, fps=fps)

        metadata_path = os.path.join(run_dir, "metadata.json")
        self.metadata.update(config)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"Video saved to {video_path}")
        print(f"Metadata saved to {metadata_path}")

        return video_frames, video_path, run_dir