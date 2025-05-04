import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image
from .base import BaseHFVideoGenerator


class CogVideoXGenerator(BaseHFVideoGenerator):
    def load_model(self):
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.model_name_or_path, torch_dtype=self.dtype
        )
        self.pipe.to("cuda")

    def optimize_pipeline(self):
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()


if __name__ == "__main__":
    prompt = "A man doing a discuss throw"
    image = load_image("data/demo/discuss_freeze_frame.png")

    generator = CogVideoXGenerator(
        model_name_or_path="THUDM/CogVideoX-5b-I2V", dtype=torch.bfloat16
    )

    config = {
        "prompt": prompt,
        "num_frames": 49,
        "num_inference_steps": 50,
        "guidance_scale": 6,
        "fps": 8,
        "seed": 42,
        "output_dir": "saved_data/cogvideox",
    }

    video_frames, video_path, frame_dir = generator.generate(
        image=image,
        config=config
    )
