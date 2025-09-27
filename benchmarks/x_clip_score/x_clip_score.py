import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from torchvision.models import vit_b_16 
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List
import argparse

MAX_LENGTH = 76
MAX_NUM_FRAMES=8

def x_clip_score(
    model, 
    tokenizer, 
    processor, 
    text:str,
    frame_path_list:List[str],
):
    
    def _read_video_frames(frame_paths, max_frames):
        total_frames = len(frame_paths)
        indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)

        selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
        return np.stack(selected_frames)
    
    input_text = tokenizer([text], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    text_feature = model.get_text_features(**input_text).flatten()

    video=_read_video_frames(frame_path_list,MAX_NUM_FRAMES)
    
    input_video = processor(videos=list(video), return_tensors="pt")
    video_feature = model.get_video_features(**input_video).flatten()
    cos_sim=F.cosine_similarity(text_feature, video_feature, dim=0).item()
    return cos_sim

def load_models(device):
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

    return model, processor, tokenizer

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
        key = class_part
        mapping[code] = key
    
    return mapping

def main(video_dir, frames_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, tokenizer = load_models(device)

    video_map = parse_filenames(video_dir)

    clip_scores = {}

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

        clip_raw_score = x_clip_score(model, tokenizer, processor, full_name, frame_files)
        
        if clip_raw_score is not None:
            clip_scores[f"{full_name}__{code}"] = float(clip_raw_score)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as f:
        json.dump(clip_scores, f, indent=4)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate CLIP Score')
    parser.add_argument('--video_dir', type=str, help='Path to the directory containing the original videos')
    parser.add_argument('--frames_dir', type=str, help='Directory where the frames are stored')
    parser.add_argument('--output_dir', type=str, help='JSON where the similarities will be stored')    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args.video_dir, args.frames_dir, args.output_dir)