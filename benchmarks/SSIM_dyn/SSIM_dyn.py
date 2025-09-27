import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from typing import List
import argparse
from skimage import io, color
from skimage.metrics import structural_similarity as ssim

DYN_SAMPLE_STEP=4

def dynamic_ssim(
    frame_path_list:List[str],
    ):
    ssim_list=[]
    sampled_list = frame_path_list[::DYN_SAMPLE_STEP]
    for f_idx in range(len(sampled_list)-1):
        frame_1=Image.open(sampled_list[f_idx])
        frame_1_gray=color.rgb2gray(frame_1)
        frame_2=Image.open(sampled_list[f_idx+1])
        frame_2_gray=color.rgb2gray(frame_2)

        data_range = frame_2_gray.max() - frame_2_gray.min()

        if data_range == 0:
            ssim_value = 1.0 
        else:
            ssim_value, _ = ssim(frame_1_gray, frame_2_gray, full=True,\
                                 data_range=data_range)
            
        ssim_list.append(ssim_value)
    ssim_avg=np.mean(ssim_list)
    
    return ssim_avg

def parse_filenames(video_dir):

    mapping = {}
    for f in os.listdir(video_dir):
        if not f.lower().endswith(".mp4"):
            continue
        if '__' not in f:
            print(f"Skipping {f} no __ found")
            continue
        
        class_part, code_ext = f.split('__', 1)
        code = code_ext[:-4]
        key = f"{class_part}__{code}"
        mapping[code] = key
    
    return mapping

def main(video_dir, frames_dir, output_dir):

    video_map = parse_filenames(video_dir)

    SSIM_dyn_scores = {}

    for code, full_name in tqdm(video_map.items()):
        frames_dir_full = os.path.join(frames_dir, code)
        print(f"Frames Directory: {frames_dir_full}")
        if not os.path.isdir(frames_dir_full):
            print(f"Frames for {code} not found")
            continue

        frame_files = sorted([os.path.join(frames_dir_full, f) for f in os.listdir(frames_dir_full) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(frame_files) == 0:
            print(f"No frames found in {frames_dir_full}. Skipping.")
            continue

        SSIM_dyn_raw_score = dynamic_ssim(frame_files)
        
        if SSIM_dyn_raw_score is not None:
            SSIM_dyn_scores[full_name] = float(SSIM_dyn_raw_score)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as f:
        json.dump(SSIM_dyn_scores, f, indent=4)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate SSIM Dynamic similarities')
    parser.add_argument('--video_dir', type=str, help='Path to the directory containing the original videos')
    parser.add_argument('--frames_dir', type=str, help='Directory where the frames are stored')
    parser.add_argument('--output_dir', type=str, help='JSON where the similarities will be stored')    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args.video_dir, args.frames_dir, args.output_dir)