import os
import random
import subprocess
import tqdm

base_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/UCF-101"
out_dir = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101"

success_count = 0
failures = []

actions = os.listdir(base_dir)

for action in actions:
    dir_path = os.path.join(base_dir, action)
    avi_files = [f for f in os.listdir(dir_path)
                 if f.endswith(".avi") and os.path.isfile(os.path.join(dir_path, f))]
    
    if len(avi_files) < 5:
        print(f"⚠️ Warning: Only {len(avi_files)} .avi files found in {dir_path}")

    selected = avi_files  # convert all files

    print(f"\n🎬 Converting {len(selected)} .avi files in action: {action}")
    action_dir = os.path.join(out_dir, action)
    os.makedirs(action_dir, exist_ok=True)

    for avi_file in tqdm.tqdm(selected, desc=f"Processing {action}"):
        input_path = os.path.join(dir_path, avi_file)
        output_name = os.path.splitext(avi_file)[0] + ".mp4"
        output_path = os.path.join(action_dir, output_name)

        result = subprocess.run(
            [
                "ffmpeg",
                "-loglevel", "error",  # show only errors
                "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            print(f"❌ Failed to convert: {input_path}")
            print(result.stderr)
            failures.append(input_path)
        else:
            success_count += 1

# Final summary
print("\n Conversion complete.")
print(f"Successful conversions: {success_count}")
print(f"Failed conversions: {len(failures)}")

if failures:
    print("\nFailed files:")
    for f in failures:
        print(f"- {f}")