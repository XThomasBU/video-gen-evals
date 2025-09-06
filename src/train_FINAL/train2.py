import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from utils import *
from losses import * 
from models import *

# set seed
torch.manual_seed(1)
np.random.seed(1)

# ——— CONFIG ———
REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh"
# ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps"]
ALL_CLASSES = ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
BATCH_SIZE = 256
LATENT_DIM = 128
EPOCHS = 200
WINDOW_SIZE = 32 # 64
STRIDE = 8 # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES"

INPUT_DIM= 1370

# ——— TRAINING ———
def train():
    print(" Loading datasets...")

    if os.path.exists(f"SAVE/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt"):
        train_samples = torch.load(f"SAVE/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_labels = torch.load(f"SAVE/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_samples = torch.load(f"SAVE/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_labels = torch.load(f"SAVE/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_vid_ids = torch.load(f"SAVE/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_vid_ids = torch.load(f"SAVE/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        train_window_ids = torch.load(f"SAVE/train_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
        test_window_ids = torch.load(f"SAVE/test_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

        # # # combine all tets values to train
        # train_samples = train_samples + test_samples
        # train_labels = train_labels + test_labels
        # train_vid_ids = train_vid_ids + test_vid_ids
        # train_window_ids = train_window_ids + test_window_ids

        
        class PoseVideoDatasetFromTensors(Dataset):
            def __init__(self, samples, labels, vid_ids, window_ids):
                self.samples = torch.stack(samples)
                self.labels = torch.tensor(labels)
                self.vid_ids = vid_ids
                self.window_ids = window_ids

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx], self.labels[idx], self.vid_ids[idx], self.window_ids[idx]


        train_dataset = PoseVideoDatasetFromTensors(train_samples, train_labels, train_vid_ids, train_window_ids)
        test_dataset = PoseVideoDatasetFromTensors(test_samples, test_labels, test_vid_ids, test_window_ids)
        print(f" Loaded existing datasets with {len(train_dataset)} train and {len(test_dataset)} test samples")
        print(f" Train samples: {train_dataset.samples.shape}, Train labels: {train_dataset.labels.shape}, Train vid_ids: {len(train_dataset.vid_ids)}, Train window_ids: {len(train_dataset.window_ids)}")
        print(f" Test samples: {test_dataset.samples.shape}, Test labels: {test_dataset.labels.shape}, Test vid_ids: {len(test_dataset.vid_ids)}, Test window_ids: {len(test_dataset.window_ids)}")
        print(" Using existing datasets for training...")
    
    # sampler = EnsurePositivesSampler(train_labels, batch_size=BATCH_SIZE, min_pos_per_class=4)
    sampler = PKBatchSampler(train_labels, P=10, K=26)
    train_loader = DataLoader(
        train_dataset,
        # batch_size=BATCH_SIZE,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(" Building model...")
    input_dim = train_dataset[0][0].shape[-1]
    model = TemporalTransformerV1(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # print number of parameters
    print(f" Number of parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = TCL()
    # loss_fn = SUPCON()
    loss_hard = SupConWithHardNegatives()
    # loss_hard = SupConHardRepelOnly()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_iter = len(train_loader) * EPOCHS
    print(f" Total iterations: {num_iter}")
    cosine_annel = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-6)

    num_classes = len(ALL_CLASSES)
    arc = ArcMarginProduct(LATENT_DIM, num_classes, s=30.0, m=0.35).to(DEVICE)
    ce = torch.nn.CrossEntropyLoss()

    all_window_ids = [w for _, _, _, w in train_dataset]
    all_vid_ids = [v for _, _, v, _ in train_dataset]

    loss_dict = {'loss': [], 'loss_con': [], 'loss_hard': [], 'smoothness': [], 'hard_shuffle_big': [], 'hard_shuffle_small': [], 'reverse': [], 'static': []}
    weights_dict = {
        "vit": [],
        "global": [],
        "pose": [],
        "beta": [],
        "kp2d": []
    }

    print(" Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_loss_con = 0
        total_loss_hard = 0
        total_smoothness_loss = 0
        total_hard_shuffle_big_loss = 0
        total_hard_shuffle_small_loss = 0
        total_reverse_loss = 0
        total_static_loss = 0
        total_weights_vit = 0
        total_weights_global = 0
        total_weights_pose = 0
        total_weights_beta = 0
        total_weights_kp2d = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            seqs, lengths, labels, vid_ids, window_ids = batch
            seqs, lengths, labels = seqs.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            shuffled_seqs = partial_shuffle_within_window(seqs, lengths, vid_ids)
            shuffled_seqs_small = partial_shuffle_within_window(seqs, lengths, vid_ids, shuffle_fraction=0.3)
            reverse_seqs = reverse_sequence(seqs, lengths)
            static_seqs = get_static_window(seqs)
            static_embeddings, _ = model(static_seqs, lengths)

            optimizer.zero_grad()
            embeddings, frame_embeddings = model(seqs, lengths)

            arc_logits = arc(embeddings, labels)                    # [B, C]
            loss_arc = ce(arc_logits, labels)
            print(loss_arc)

            # weights = model.last_mod_weights   # [B, M]
            # modalities = model.modalities      # ["vit", "global", "pose", "beta", "kp2d"]

            # ent = -(weights * (weights.clamp_min(1e-8)).log()).sum(dim=-1).mean() 

            # mean_weights = weights.mean(dim=0)  # [M]
            # # for m_idx, m in enumerate(modalities):
            # #     print(f"{m}: {mean_weights[m_idx].item():.4f}")
            # total_weights_vit += mean_weights[0].item()
            # total_weights_global += mean_weights[1].item()
            # total_weights_pose += mean_weights[2].item()
            # total_weights_beta += mean_weights[3].item()
            # total_weights_kp2d += mean_weights[4].item()

            # with torch.no_grad():
            shuffled_embeddings, _        = model(shuffled_seqs,   lengths)
            shuffled_embeddings_small, _  = model(shuffled_seqs_small, lengths)
            reverse_embeddings, _         = model(reverse_seqs,        lengths)
            static_embeddings, _          = model(static_seqs,         lengths)

            # current_window_embeddings = []
            # next_window_embeddings = []

            # for i in range(len(vid_ids)):
            #     next_idx = get_next_window_index(all_vid_ids, all_window_ids, vid_ids[i], window_ids[i])
            #     if next_idx is not None:
            #         # Get the next window's input (you might want to cache/preload these for speed!)
            #         next_seq, next_length, _, _ = train_dataset[next_idx]
            #         next_seq = next_seq.unsqueeze(0).to(DEVICE)
            #         next_length = torch.tensor([next_length]).to(DEVICE)
            #         with torch.no_grad():
            #             next_emb, _ = model(next_seq, next_length)
            #         current_window_embeddings.append(embeddings[i])         # embeddings for the current batch
            #         next_window_embeddings.append(next_emb.squeeze(0))      # next window's embedding
            # if current_window_embeddings:  # Only compute if non-empty
            #     current_window_embeddings = torch.stack(current_window_embeddings)  # [N, D]
            #     next_window_embeddings = torch.stack(next_window_embeddings)  # [N, D]
            #     l2_smoothness_loss = (current_window_embeddings - next_window_embeddings).norm(dim=1).mean()

            # get cosine sim 
            cosine_sim = torch.cosine_similarity(embeddings.unsqueeze(1), shuffled_embeddings.unsqueeze(0), dim=-1)  # [B, B]         
            loss_org = loss_fn(embeddings, labels)

            # # # ——— temporal smoothness penalty ———
            # # # build a map: vid_id → list of batch‐indices
            # vid_to_indices = defaultdict(list)
            # for i, vid in enumerate(vid_ids):
            #     vid_to_indices[vid].append(i)

            shuffled_big_loss = loss_hard(embeddings, embeddings, shuffled_embeddings)
            shuffled_small_loss = loss_hard(embeddings, embeddings, shuffled_embeddings_small)
            reverse_loss = loss_hard(embeddings, embeddings, reverse_embeddings)
            static_loss = loss_hard(embeddings, embeddings, static_embeddings)
            # print(shuffled_big_loss.item(), shuffled_small_loss.item(), reverse_loss.item(), static_loss.item())

            # Combine hard losses
            loss_hard_combined = shuffled_big_loss + reverse_loss + static_loss
            # print(cosine_sim.mean().item(), loss_org.item(), loss_hard_combined.item())

            # frame smoothness loss
            frame_smoothness_loss = second_order_steady_loss(frame_embeddings[:, 1:])  # exclude CLS token
            # frame_smoothness_loss = second_order_steady_loss(frame_embeddings)  # exclude CLS token

            # print(ent)

            loss = loss_org +  10 * loss_hard_combined + loss_arc * 0.1
            # loss = loss_org +  10 * loss_hard_combined
            # loss = loss_org
            # loss = loss_org + 10 * loss_hard_combined
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_con += loss_org.item()
            total_loss_hard += loss_hard_combined.item()
            total_smoothness_loss += frame_smoothness_loss.item()
            total_hard_shuffle_big_loss += shuffled_big_loss.item()
            total_hard_shuffle_small_loss += shuffled_small_loss.item()
            total_reverse_loss += reverse_loss.item()
            total_static_loss += static_loss.item()
            # cosine_annel.step()
        print(f" Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        loss_dict['loss'].append(total_loss / len(train_loader))
        loss_dict['loss_con'].append(total_loss_con / len(train_loader))
        loss_dict['loss_hard'].append(total_loss_hard / len(train_loader))
        loss_dict['smoothness'].append(total_smoothness_loss / len(train_loader))
        loss_dict['hard_shuffle_big'].append(total_hard_shuffle_big_loss / len(train_loader))
        loss_dict['hard_shuffle_small'].append(total_hard_shuffle_small_loss / len(train_loader))
        loss_dict['reverse'].append(total_reverse_loss / len(train_loader))
        loss_dict['static'].append(total_static_loss / len(train_loader))

        # weights_dict["vit"].append(total_weights_vit / len(train_loader))
        # weights_dict["global"].append(total_weights_global / len(train_loader))
        # weights_dict["pose"].append(total_weights_pose / len(train_loader))
        # weights_dict["beta"].append(total_weights_beta / len(train_loader))
        # weights_dict["kp2d"].append(total_weights_kp2d / len(train_loader))

    print(" Saving model...")
    torch.save(model.state_dict(), f"SAVE/temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt")
    # model.load_state_dict(torch.load(f"SAVE/temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt", map_location=DEVICE))

    post_training(loss_dict, model, train_loader, test_loader, ALL_CLASSES, WINDOW_SIZE, STRIDE, DEVICE)

    # results = test_embedding_sensitivity(model, test_loader, centroids, ALL_CLASSES, DEVICE)

# ——— MAIN ———
if __name__ == "__main__":
    train()