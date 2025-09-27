import os
import re
import json
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import vit_b_16 
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List
import argparse

MAX_LENGTH = 76

def clip_score(
    model, 
    tokenizer,
    text:str,
    frame_path_list:List[str],
):
    device=model.device
    input_t = tokenizer(text=text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", padding=True).to(device)
    cos_sim_list=[]
    for frame_path in frame_path_list:
        image=Image.open(frame_path)
        input_f = tokenizer(images=image, return_tensors="pt", padding=True).to(device)
        output_t = model.get_text_features(**input_t).flatten()
        output_f = model.get_image_features(**input_f).flatten()
        cos_sim = F.cosine_similarity(output_t, output_f, dim=0).item()
        cos_sim_list.append(cos_sim)
    clip_score_avg=np.mean(cos_sim_list)

    return clip_score_avg

def load_models(device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return clip_model, clip_processor

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
        key = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', class_part).strip()
        mapping[code] = key
    
    return mapping

def main(video_dir, frames_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_models(device)

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

        clip_raw_score = clip_score(model, processor, full_name, frame_files)
        
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