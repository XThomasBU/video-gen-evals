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

def dino_inter_frame(
    model, 
    preprocess, 
    frame_path_list:List[str],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_sim_list=[]
    for f_idx in tqdm(range(len(frame_path_list)-1)):
        frame_1=Image.open(frame_path_list[f_idx])
        frame_2=Image.open(frame_path_list[f_idx+1])
        frame_tensor_1 = preprocess(frame_1).unsqueeze(0).to(device)
        frame_tensor_2 = preprocess(frame_2).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_1 = model(frame_tensor_1).flatten()
            feat_2 = model(frame_tensor_2).flatten()
        cos_sim=F.cosine_similarity(feat_1, feat_2, dim=0).item()
        frame_sim_list.append(cos_sim)
    frame_sim_avg=np.mean(frame_sim_list)
    return frame_sim_avg

def load_models(device):
    model = vit_b_16(pretrained=True)
    model.to(device)
    model.eval()  
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    

    return model, preprocess

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessor = load_models(device)

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

        dino_raw_score = dino_inter_frame(model, preprocessor, frame_files)
        
        if dino_raw_score is not None:
            clip_scores[full_name] = float(dino_raw_score)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as f:
        json.dump(clip_scores, f, indent=4)
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate DINO cosine similarities')
    parser.add_argument('--video_dir', type=str, help='Path to the directory containing the original videos')
    parser.add_argument('--frames_dir', type=str, help='Directory where the frames are stored')
    parser.add_argument('--output_dir', type=str, help='JSON where the similarities will be stored')    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args.video_dir, args.frames_dir, args.output_dir)