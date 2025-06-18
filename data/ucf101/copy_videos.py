import os
import shutil

txt_file = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/data/ucf101/filter.txt"
prefix_to_remove = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/"
base_dst_root = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_10classes"

with open(txt_file, 'r') as f:
    video_paths = [line.strip() for line in f if line.strip()]

for full_path in video_paths:
    if not full_path.startswith(prefix_to_remove):
        continue
    rel_path = full_path[len(prefix_to_remove):]
    rel_path_new = rel_path.replace("ucf101/", "ucf101_10classes/", 1)
    dst_path = os.path.join(prefix_to_remove, rel_path_new)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(full_path, dst_path)