import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from pathlib import Path
import numpy as np

# -----------------------------
# Load your model + tensors
# -----------------------------
from models import TemporalTransformerV2Plus
from train import (
    ALL_CLASSES,
    WINDOW_SIZE, STRIDE, LATENT_DIM,
    DEVICE, INPUT_DIM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Loading model...")
model = TemporalTransformerV2Plus(input_dim=INPUT_DIM*2, latent_dim=LATENT_DIM).to(DEVICE)
ckpt = f"SAVE/temporal_transformer_model_window_32_stride_8_valid_window_NO_ENT.pt"
print(f" Loading weights from {ckpt}")
state = torch.load(ckpt, map_location="cpu")
model.load_state_dict(state)
model.eval()

print(f" #params: {sum(p.numel() for p in model.parameters()):,}")

# ---- embeddings / labels / ids
all_train_embeds = torch.load(f"SAVE/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt", map_location="cpu")
all_train_labels = torch.load(f"SAVE/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt", map_location="cpu")
all_train_vid_ids = torch.load(f"SAVE/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window_NO_ENT.pt", map_location="cpu", weights_only=False)

train_samples = torch.load(f"SAVE/train_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt", map_location="cpu")
train_labels  = torch.load(f"SAVE/train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt",  map_location="cpu")
test_samples  = torch.load(f"SAVE/test_samples_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt",  map_location="cpu")
test_labels   = torch.load(f"SAVE/test_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt",   map_location="cpu")
train_vid_ids = torch.load(f"SAVE/train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt", map_location="cpu")
test_vid_ids  = torch.load(f"SAVE/test_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt",  map_location="cpu")
train_window_ids = torch.load(f"SAVE/train_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt", map_location="cpu")
test_window_ids  = torch.load(f"SAVE/test_window_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt",  map_location="cpu")

# -----------------------------
# Dataset wrapper (from tensors)
# -----------------------------
class PoseVideoDatasetFromTensors(Dataset):
    def __init__(self, samples, labels, vid_ids, window_ids):
        # stack to a single tensor once
        self.samples = torch.stack(list(samples)) if isinstance(samples, (list, tuple)) else samples
        self.labels = torch.tensor(labels) if not torch.is_tensor(labels) else labels
        self.vid_ids = np.array(vid_ids) if not isinstance(vid_ids, np.ndarray) else vid_ids
        self.window_ids = np.array(window_ids) if not isinstance(window_ids, np.ndarray) else window_ids

    def __len__(self):
        return int(self.samples.shape[0])

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.vid_ids[idx], self.window_ids[idx]

data = PoseVideoDatasetFromTensors(train_samples, train_labels, train_vid_ids, train_window_ids)
loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=2, pin_memory=(DEVICE.type=="cuda"), drop_last=False)

# -----------------------------
# Class prototypes (video means -> class mean)
# -----------------------------
# (If you already saved prototypes, feel free to load instead.)
centroids = {}
classes_tensor = torch.unique(all_train_labels)
for cls in classes_tensor:
    mask_cls = (all_train_labels == cls)
    embeds_cls = all_train_embeds[mask_cls]  # [N_c, D]
    vid_ids_cls = np.array(all_train_vid_ids)[mask_cls.numpy()]
    video_means = []
    for vid in np.unique(vid_ids_cls):
        vmask = (vid_ids_cls == vid)
        video_means.append(embeds_cls[vmask].mean(dim=0))
    centroids[int(cls.item())] = torch.stack(video_means, dim=0).mean(dim=0)  # [D]

# -----------------------------
# Modality slicing (derived from model)
# -----------------------------
MOD_DIMS = dict(model.dim_map)            # e.g., {"vit":1024,"global":9,"pose":207,"beta":10,"kp2d":120}
MOD_ORDER = ["vit", "global", "pose", "beta", "kp2d"]
ONE_PASS_DIM = sum(MOD_DIMS[m] for m in MOD_ORDER)
HAS_DIFF = (model.total_in > ONE_PASS_DIM)

def build_slices():
    s = {}
    start = 0
    for m in MOD_ORDER:
        d = MOD_DIMS[m]
        raw  = slice(start, start+d)
        diff = None
        if HAS_DIFF:
            diff = slice(ONE_PASS_DIM + start, ONE_PASS_DIM + start + d)
        s[m] = (raw, diff)
        start += d
    return s

SLICES = build_slices()

# -----------------------------
# Interventions
# -----------------------------
@torch.no_grad()
def intervene(seqs, modality, kind="zero", same_noise=False):
    """
    seqs: [B, T, total_in]
    kind: 'zero', 'freeze_motion', 'freeze_state', 'zero_raw', 'zero_diff', 'noise', 'shuffle_time'
    """
    x = seqs.clone()
    raw_sl, diff_sl = SLICES[modality]
    B, T, D = x.shape
    dev = x.device

    if kind in ["zero", "freeze_state"] and raw_sl is not None:
        x[:, :, raw_sl] = 0.0
    if kind in ["zero", "freeze_motion", "zero_diff"] and diff_sl is not None:
        x[:, :, diff_sl] = 0.0
    if kind == "zero_raw" and raw_sl is not None:
        x[:, :, raw_sl] = 0.0

    if kind == "noise":
        if raw_sl is not None:
            shape_r = (1,1,raw_sl.stop-raw_sl.start) if same_noise else (B,T,raw_sl.stop-raw_sl.start)
            x[:, :, raw_sl] = torch.randn(*shape_r, device=dev)
        if diff_sl is not None:
            shape_d = (1,1,diff_sl.stop-diff_sl.start) if same_noise else (B,T,diff_sl.stop-diff_sl.start)
            x[:, :, diff_sl] = torch.randn(*shape_d, device=dev)

    if kind == "shuffle_time":
        for b in range(B):
            perm = torch.randperm(T, device=dev)
            if raw_sl is not None:
                x[b, :, raw_sl]  = x[b, perm, :][:, raw_sl]
            if diff_sl is not None:
                x[b, :, diff_sl] = x[b, perm, :][:, diff_sl]
    return x

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_modality_impact_streaming(
    model, loader, centroids, device,
    kinds=("zero","freeze_motion","freeze_state"),
    use_fp16=True
):
    model.eval()

    # Prototypes [C, D] normalized
    classes = sorted(centroids.keys())
    Proto = torch.stack([centroids[c] for c in classes], dim=0).to(device)
    Proto = F.normalize(Proto, dim=-1)

    def dist_to_own(z, y):
        own = Proto[y]                  # [B, D]
        return torch.norm(z - own, dim=-1)

    agg = defaultdict(lambda: defaultdict(lambda: {
        "cos_drop_sum":0.0, "intra_delta_sum":0.0, "flip_count":0, "N":0,
        "per_class": defaultdict(lambda: {"cos_drop_sum":0.0, "intra_delta_sum":0.0, "flip_count":0, "N":0})
    }))

    if device.type == "cuda" and use_fp16:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, a,b,c): return False
        autocast_ctx = DummyCtx()

    with torch.no_grad(), autocast_ctx:
        for seqs, labels, *_ in loader:
            seqs   = seqs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            # baseline
            z_base, _, _ = model(seqs)
            z_base = F.normalize(z_base, dim=-1)

            intra_base = dist_to_own(z_base, labels)
            sim_base = z_base @ Proto.T
            nb = sim_base.argmax(dim=1)

            for m in MOD_ORDER:
                for kind in kinds:
                    seqs_abl = intervene(seqs, m, kind=kind)
                    z_abl, _, _ = model(seqs_abl)
                    z_abl = F.normalize(z_abl, dim=-1)

                    cos = (z_base * z_abl).sum(dim=-1).clamp(-1, 1)          # [B]
                    cos_drop = (1.0 - cos)

                    intra_abl = dist_to_own(z_abl, labels)
                    intra_delta = (intra_abl - intra_base)

                    sim_abl = z_abl @ Proto.T
                    na = sim_abl.argmax(dim=1)
                    flips = (nb != na).to(torch.int32)

                    rec = agg[m][kind]
                    rec["cos_drop_sum"]    += float(cos_drop.sum().item())
                    rec["intra_delta_sum"] += float(intra_delta.sum().item())
                    rec["flip_count"]      += int(flips.sum().item())
                    rec["N"]               += int(labels.numel())

                    # per-class aggregates
                    for c in torch.unique(labels):
                        cmask = (labels == c)
                        Nc = int(cmask.sum().item())
                        if Nc == 0: continue
                        pr = rec["per_class"][int(c.item())]
                        pr["cos_drop_sum"]    += float(cos_drop[cmask].sum().item())
                        pr["intra_delta_sum"] += float(intra_delta[cmask].sum().item())
                        pr["flip_count"]      += int(flips[cmask].sum().item())
                        pr["N"]               += Nc

            # free batch refs
            del seqs, labels, z_base, intra_base, sim_base, nb

    # finalize + print
    results = defaultdict(dict)
    for m in MOD_ORDER:
        for kind in kinds:
            rec = agg[m][kind]
            N = max(rec["N"], 1)
            gm = {
                "cos_drop":    rec["cos_drop_sum"] / N,
                "intra_delta": rec["intra_delta_sum"] / N,
                "flip_rate":   rec["flip_count"]   / N,
            }
            cm = {}
            for c, pr in rec["per_class"].items():
                Nc = max(pr["N"], 1)
                cm[c] = {
                    "cos_drop":    pr["cos_drop_sum"] / Nc,
                    "intra_delta": pr["intra_delta_sum"] / Nc,
                    "flip_rate":   pr["flip_count"]    / Nc,
                }
            results[m][kind] = {"global": gm, "per_class": cm}
            print(f"[{m:6s} | {kind:12s}]  cos_drop={gm['cos_drop']:.4f}  intraΔ={gm['intra_delta']:.4f}  flip={100*gm['flip_rate']:.2f}%")

    return results

# -----------------------------
# Run eval
# -----------------------------
results = evaluate_modality_impact_streaming(
    model, loader, centroids, DEVICE,
    kinds=("zero","freeze_motion","freeze_state"),
    use_fp16=True
)