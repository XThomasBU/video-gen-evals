import os
import random
import subprocess

directories = [
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/WallPushups",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/PushUps",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/Punch",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/PullUps",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/PlayingTabla",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/JumpRope",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/JumpingJack",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/HulaHoop",
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101/BoxingSpeedBag",
]

for dir_path in directories:
    avi_files = [f for f in os.listdir(dir_path) if f.endswith(".avi")]
    if len(avi_files) < 5:
        print(f"Warning: Only {len(avi_files)} .avi files found in {dir_path}")
    selected = random.sample(avi_files, min(5, len(avi_files)))

    out_dir = os.path.join(dir_path, "converted_mp4")
    os.makedirs(out_dir, exist_ok=True)

    for avi_file in selected:
        input_path = os.path.join(dir_path, avi_file)
        output_name = os.path.splitext(avi_file)[0] + ".mp4"
        output_path = os.path.join(out_dir, output_name)

        print(f"Converting {avi_file} -> {output_name}")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                output_path,
            ],
            check=True,
        )

print("All videos converted.")
