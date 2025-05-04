import torch
from diffusers.utils import export_to_video
import os
from datetime import datetime
import json
from runwayml import RunwayML
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO

load_dotenv()


class BaseHFVideoGenerator:
    def __init__(self, model_name_or_path, dtype=torch.float16, output_root="outputs"):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.pipe = None
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        self.metadata = {"model_name": model_name_or_path, "output_root": output_root}

    def load_model(self):
        raise NotImplementedError

    def optimize_pipeline(self):
        pass

    def _create_unique_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_root, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def generate(self, image, config: dict):
        if self.pipe is None:
            self.load_model()
            self.optimize_pipeline()

        # Extract only what's relevant for this model
        prompt = config["prompt"]
        num_frames = config.get("num_frames", 24)
        num_inference_steps = config.get("num_inference_steps", 50)
        guidance_scale = config.get("guidance_scale", 7)
        fps = config.get("fps", 8)
        seed = config.get("seed", 42)
        output_dir = config.get("output_dir", None)

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

        run_dir = output_dir or self._create_unique_output_dir()
        os.makedirs(run_dir, exist_ok=True)

        frame_dir = os.path.join(run_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))

        video_path = os.path.join(run_dir, "video.mp4")
        export_to_video(video_frames, video_path, fps=fps)

        # Save metadata
        metadata_path = os.path.join(run_dir, "metadata.json")
        self.metadata.update(config)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"Video saved to {video_path}")
        print(f"Metadata saved to {metadata_path}")

        return video_frames, video_path, frame_dir


class BaseRunwayVideoGenerator:
    def __init__(self, model_name, output_root="outputs"):
        self.model_name = model_name
        self.output_root = output_root
        self.client = None
        os.makedirs(self.output_root, exist_ok=True)

        self.metadata = {"model_name": model_name, "output_root": output_root}

    def load_model(self):
        api_key = os.getenv("RUNWAYML_API_SECRET")
        if api_key is None:
            raise ValueError(
                "RUNWAYML_API_SECRET not found in environment variables or .env file"
            )
        self.client = RunwayML(api_key=api_key)

    def _create_unique_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_root, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _convert_image_to_data_uri(self, path):
        if path.startswith("http://") or path.startswith("https://"):
            return path
        else:
            with Image.open(path) as img:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
            b64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/png;base64,{b64_encoded}"

    def generate(self, config: dict):
        if self.client is None:
            self.load_model()

        prompt = config["prompt"]
        image_path = config["image_path"]
        fps = config.get("fps", 8)
        output_dir = config.get("output_dir", None)

        prompt_image = self._convert_image_to_data_uri(image_path)

        result = self.client.image_to_video.create(
            model=self.model_name,
            prompt_image=prompt_image,
            ratio="1280:720",
            prompt_text=prompt,
        )

        video_id = result.id
        video_url = result.urls.video

        run_dir = output_dir or self._create_unique_output_dir()
        os.makedirs(run_dir, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(run_dir, "metadata.json")
        self.metadata.update(config)
        self.metadata.update(
            {
                "video_id": video_id,
                "video_url": video_url,
            }
        )
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"Video URL: {video_url}")
        print(f"Metadata saved to {metadata_path}")

        return video_url, run_dir
