import torch
from diffusers.utils import export_to_video
import os
from datetime import datetime
import json

class BaseHFVideoGenerator:
    def __init__(self, model_name_or_path, dtype=torch.float16, output_root="outputs"):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.pipe = None
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        self.metadata = {
            "model_name": model_name_or_path,
            "output_root": output_root
        }

    def load_model(self):
        """Override in subclass to load the specific Hugging Face pipeline."""
        raise NotImplementedError

    def optimize_pipeline(self):
        """Optional: Override if the pipeline supports tiling, slicing, etc."""
        pass

    def _create_unique_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_root, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def generate(
        self,
        prompt,
        image,
        num_frames=24,
        num_inference_steps=50,
        guidance_scale=7,
        fps=8,
        seed=42,
        output_dir=None,
    ):
        if self.pipe is None:
            self.load_model()
            self.optimize_pipeline()

        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        video_frames = result.frames[0]

        if output_dir is None:
            # Create unique output directory
            run_dir = self._create_unique_output_dir()
        else:
            run_dir = output_dir
            os.makedirs(run_dir, exist_ok=True)

        # Save frames
        frame_dir = os.path.join(run_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))

        # Save video
        video_path = os.path.join(run_dir, "video.mp4")
        export_to_video(video_frames, video_path, fps=fps)

        print(f"Video saved to {video_path}")
        print(f"Frames saved to {frame_dir}")

        # save metadata file
        metadata_path = os.path.join(run_dir, "metadata.json")
        self.metadata["prompt"] = prompt
        self.metadata["num_frames"] = num_frames
        self.metadata["num_inference_steps"] = num_inference_steps
        self.metadata["guidance_scale"] = guidance_scale
        self.metadata["fps"] = fps
        self.metadata["seed"] = seed

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")


        return video_frames, video_path, frame_dir