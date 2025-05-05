import os
from runwayml import RunwayML
from .base import BaseRunwayVideoGenerator


class RunwayGen4TurboGenerator(BaseRunwayVideoGenerator):
    def load_model(self):
        self.client = RunwayML(api_key=os.getenv("RUNWAYML_API_SECRET"))


if __name__ == "__main__":
    prompt = "A man doing a discuss throw"
    image_url = "data/demo/discuss_freeze_frame.png"

    generator = RunwayGen4TurboGenerator(model_name="gen4_turbo")
    config = {
        "model_name": "gen4_turbo",
        "output_root": "outputs",
        "image_path": image_url,
        "duration": 5,
    }

    video_url, local_path = generator.generate(
        prompt=prompt,
        config=config,
    )
