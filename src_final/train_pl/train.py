

import os
import sys
import json
import math
import random
import typing as T
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------- local project imports ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import TemporalTransformerV2Plus
from losses import *
from utils import *  # partial_shuffle_within_window, reverse_sequence, get_static_window, etc.

# --------------------------- Determinism ---------------------------

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator(); g.manual_seed(SEED)

# ------------------------------ Config -----------------------------

DATASET_DIR = "/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz"
# Expect layout: DATASET_DIR/<class>/<video>.npz produced by save_video_npz(...)

# WHITELIST_JSON_DIR ="/home/coder/projects/video_evals/video-gen-evals/FINAL_MESH_UCF101/single"
FILTER_CLASSES_TEST =  ["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"]
WHITELIST_JSON_DIR = None
FILTER_CLASSES = None

BATCH_SIZE = 2048
LATENT_DIM = 128
EPOCHS = 90
CLIP_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NGPUS = torch.cuda.device_count()
USE_DP = (DEVICE == "cuda") and (NGPUS > 1)
PRIMARY_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")

TOTAL_WINDOWS_PER_EPOCH = 16384
WINDOWS_PER_VIDEO = 8
STRIDE = 8

RETRIES = 2          # retries are cheaper here (arrays); 2 is usually enough
JITTER = CLIP_LEN // 2
PAD_MODE = "repeat"  # repeat nearest frame if window lands near edges


# ============================== Training ==============================

# Build dataset + splits
full_ds = NpzVideoDataset(DATASET_DIR, whitelist_json_dir=WHITELIST_JSON_DIR, filter_classes=FILTER_CLASSES)
train_ds, test_ds = train_test_split(full_ds, train_ratio=0.8, seed=SEED)
stats = compute_stats_from_npz(train_ds.items)
print(f"Train videos: {len(train_ds)}, Test videos: {len(test_ds)}")

# After split
from collections import Counter

def class_counts(ds):
    return Counter([it.cls for it in ds.items])

print("FULL:", {k: len(v) for k, v in full_ds.class_to_items.items()})
print("TRAIN:", class_counts(train_ds))
print("TEST:", class_counts(test_ds))

# Ensure every class in FULL appears in both TRAIN and TEST
full_classes = set(full_ds.class_to_items.keys())
assert full_classes.issubset(set(train_ds.classes) | set(test_ds.classes))
assert all(c in train_ds.classes for c in full_classes)
assert all(c in test_ds.classes  for c in full_classes)

ALL_CLASSES = sorted(list({it.cls for it in full_ds.items}))
label_dict = {cls: i for i, cls in enumerate(ALL_CLASSES)}

# save label mapping
os.makedirs("SAVE", exist_ok=True)
with open("SAVE/label_mapping.json", "w") as f:
    json.dump(label_dict, f, indent=2)

# Peek one window to determine feature dim
probe_samples = sample_windows_capped_npz(train_ds, clip_len=CLIP_LEN, stride=STRIDE,
                                          windows_per_video=1, total_cap=1, seed=SEED)
assert len(probe_samples) > 0, "No training windows found."
probe_ds = WindowDataset(probe_samples, clip_len=CLIP_LEN, retries=0, jitter=0, pad_mode=PAD_MODE, stats=stats, seed=SEED)
probe = probe_ds[0]
assert probe is not None
probe_feats, _, _ = probe
INPUT_DIM = probe_feats.shape[-1]
print(f"Derived input_dim per frame = {INPUT_DIM}")

# Create model/head on primary device
model = TemporalTransformerV2Plus(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(PRIMARY_DEVICE)
arc = ArcMarginProduct(LATENT_DIM, len(ALL_CLASSES), s=30.0, m=0.35).to(PRIMARY_DEVICE)
ce = torch.nn.CrossEntropyLoss().to(PRIMARY_DEVICE)

if USE_DP:
    print(f"Using DataParallel across {NGPUS} GPUs: {list(range(NGPUS))}")
    model = torch.nn.DataParallel(model, device_ids=list(range(NGPUS)))
    arc = torch.nn.DataParallel(arc, device_ids=list(range(NGPUS)))

loss_fn = TCL().to(PRIMARY_DEVICE)
loss_hard = SupConWithHardNegatives().to(PRIMARY_DEVICE)

params = list(model.parameters()) + list(arc.parameters())
optimizer = torch.optim.AdamW(params, lr=3e-4)
# Per-step cosine: approximate steps/epoch from TOTAL_WINDOWS_PER_EPOCH / BATCH_SIZE
steps_per_epoch = max(1, math.ceil(TOTAL_WINDOWS_PER_EPOCH / max(1, BATCH_SIZE)))
cosine_anneal = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=steps_per_epoch * EPOCHS, eta_min=1e-6
)

small_train_loader = make_stratified_train_loader(
    train_ds, clip_len=CLIP_LEN, stride=STRIDE, pad_mode=PAD_MODE, stats=stats,
    K_per_class=64, windows_per_video=2, seed=SEED, batch_size=256
)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    samples = sample_windows_capped_npz(
        train_ds,
        clip_len=CLIP_LEN,
        stride=STRIDE,
        windows_per_video=WINDOWS_PER_VIDEO,
        total_cap=TOTAL_WINDOWS_PER_EPOCH,
        seed=SEED + epoch
    )
    print(f"Sampled {len(samples)} windows for this epoch.")

    window_ds = WindowDataset(
        samples, clip_len=CLIP_LEN, retries=RETRIES, jitter=JITTER, pad_mode=PAD_MODE, seed=SEED + epoch, stats=stats
    )

    # train_loader = DataLoader(
    #     window_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=0,               # safe default on shared clusters
    #     worker_init_fn=seed_worker,
    #     generator=g,
    #     collate_fn=safe_collate,
    #     pin_memory=(DEVICE == "cuda"),
    # )
    P, K = 10, 24   # => batch size 256 (independent of BATCH_SIZE var)
    labels_for_sampler = [label_dict[it.cls] for (it, _s) in window_ds.samples]
    pk_sampler = PKBatchSampler(labels_for_sampler, P=P, K=K, drop_last=True)

    train_loader = DataLoader(
        window_ds,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=safe_collate,
        pin_memory=(DEVICE == "cuda"),
        batch_sampler=pk_sampler,          # <- use sampler
    )

    test_loader = make_test_loader(
        test_ds, clip_len=CLIP_LEN, stride=STRIDE,
        pad_mode=PAD_MODE, stats=stats, seed=SEED, batch_size=256, filter_classes=FILTER_CLASSES_TEST
    )
    test_loader_full = make_test_loader(
        test_ds, clip_len=CLIP_LEN, stride=STRIDE,
        pad_mode=PAD_MODE, stats=stats, seed=SEED, batch_size=256
    )

    # train_eval_loader = make_test_loader(
    #     train_ds,
    #     clip_len=CLIP_LEN,
    #     stride=STRIDE,
    #     pad_mode=PAD_MODE,
    #     stats=stats,
    #     seed=SEED,
    #     batch_size=256
    # )

    best_score = -1.0

    total_loss, steps = 0.0, 0
    for packed in tqdm(train_loader, desc=f"Train (epoch {epoch+1})", leave=False):
        if packed is None:
            continue
        feats, cls_names, vids = packed

        labels = torch.as_tensor([label_dict[c] for c in cls_names], dtype=torch.long)

        # Augment on CPU, then move once
        shuffled_feats = partial_shuffle_within_window(feats, shuffle_fraction=0.7)
        reverse_feats  = reverse_sequence(feats)
        static_feats   = get_static_window(feats)

        feats          = feats.to(PRIMARY_DEVICE, non_blocking=True)
        shuffled_feats = shuffled_feats.to(PRIMARY_DEVICE, non_blocking=True)
        reverse_feats  = reverse_feats.to(PRIMARY_DEVICE, non_blocking=True)
        static_feats   = static_feats.to(PRIMARY_DEVICE, non_blocking=True)
        labels         = labels.to(PRIMARY_DEVICE, non_blocking=True)

        optimizer.zero_grad()

        emb, _, _   = model(feats)
        sh_emb,_,_  = model(shuffled_feats)
        rev_emb,_,_ = model(reverse_feats)
        st_emb,_,_  = model(static_feats)

        logits_arc = arc(emb, labels)                             # [B, C]
        loss_arc = ce(logits_arc, labels) 

        loss_org = loss_fn(emb, labels)  +  action_consistency_loss(emb, labels, ratio=True) + (
            loss_hard(emb, emb, sh_emb) +
            loss_hard(emb, emb, rev_emb) +
            loss_hard(emb, emb, st_emb)
        ) + 0.1 * loss_arc 
        # loss = loss_org + 10.0 * (
        #     loss_hard(emb, emb, sh_emb) +
        #     loss_hard(emb, emb, rev_emb) +
        #     loss_hard(emb, emb, st_emb)
        # ) + 0.1 * loss_arc +  action_consistency_loss(emb, labels, ratio=True)

        loss = loss_org

        if not torch.isfinite(loss):
            print("⚠️ Non-finite loss, skipping batch.",
                  f"loss_org={loss_org.item() if torch.isfinite(loss_org) else 'nan'}")
            continue

        loss.backward()
        optimizer.step()
        # cosine_anneal.step()

        total_loss += loss.item()
        # # steps += 1
        # # if steps % 10 == 0:
        # print(f"Step {steps}, Loss: {loss.item():.4f} (Org: {loss_org.item():.4f})")

    # # save model checkpoint
    # os.makedirs("SAVE", exist_ok=True)
    # torch.save(model.state_dict(), f"SAVE/temporal_transformer_epoch{epoch+1:03d}.pt")

    avg_loss = total_loss / len(train_loader) 

    centroids_sub, counts_sub = build_train_centroids_subset(
        model, small_train_loader, label_dict, device=PRIMARY_DEVICE
    )

    vid_stats, cls_stats, avg_sc, max_sc, min_sc, med_sc = action_consistency_centroid(
        model, test_loader, label_dict, centroids_sub, device=PRIMARY_DEVICE
    )
    num_ge_09 = sum(1 for v in cls_stats.values() if np.isfinite(v["avg"]) and v["avg"] >= 0.9)
    print(f"[Subset-Centroid AC] avg {avg_sc:.4f} | max {max_sc:.4f} | min {min_sc:.4f} | median {med_sc:.4f} | classes >= 0.9: {num_ge_09}/{len(cls_stats)}")

    vid_stats_full, cls_stats_full, avg_sc_full, max_sc_full, min_sc_full, med_sc_full = action_consistency_centroid(
        model, test_loader_full, label_dict, centroids_sub, device=PRIMARY_DEVICE
    )
    num_ge_09_full = sum(1 for v in cls_stats_full.values() if np.isfinite(v["avg"]) and v["avg"] >= 0.9)
    print(f"[Full-Centroid AC]   avg {avg_sc_full:.4f} | max {max_sc_full:.4f} | min {min_sc_full:.4f} | median {med_sc_full:.4f} | classes >= 0.9: {num_ge_09_full}/{len(cls_stats_full)}")

    # per_video_stats, avg_score, max_score, min_score, median_score = action_consistency(model, test_loader, label_dict, device=PRIMARY_DEVICE)
    # per_class_scores = defaultdict(list)

    # for vid, d in per_video_stats.items():
    #     cls_id = d["class"]
    #     per_class_scores[cls_id].append(d["score"])

    # class_stats = {
    #     cls_id: {
    #         "count_videos": len(scores),
    #         "avg": float(np.nanmean(scores)) if scores else float("nan"),
    #         "max": float(np.nanmax(scores))  if scores else float("nan"),
    #         "min": float(np.nanmin(scores))  if scores else float("nan"),
    #         "median": float(np.nanmedian(scores)) if scores else float("nan"),
    #     }
    #     for cls_id, scores in per_class_scores.items()
    # }

    # # If you really mean “classes with score > 0.9” based on class-average:
    # classes_with_score_greater_than_0_9 = sum(
    #     1 for s in class_stats.values() if np.isfinite(s["avg"]) and s["avg"] > 0.9
    # )
    # print(f"Epoch {epoch+1}: avg train loss {avg_loss:.4f} | avg score {avg_score:.4f} | max score {max_score:.4f} | min score {min_score:.4f} | median score {median_score:.4f} | num classes >= 0.9: {classes_with_score_greater_than_0_9} / {len(ALL_CLASSES)}")
    # # # (optional) keep best checkpoint by consistency
    # if np.isfinite(avg_score) and avg_score > best_score:
    #     best_score = avg_score
    #     torch.save(model.state_dict(), f"SAVE/temporal_transformer_best.pt")

print("✅ Training complete.")
