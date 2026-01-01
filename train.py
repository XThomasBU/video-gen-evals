import os, sys, json, random, datetime, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use local tmp to avoid NFS .nfs* cleanup errors from multiprocessing
os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = os.environ["TMPDIR"]

GLOBAL_CONFIG = {
    "seed": 1337,
    "use_dataparallel": False,
    "paths": {
        "mesh_human_data_generated": "generated_meshes",
        "mesh_human_data_real": "meshes_10classes",
        "human_scores": "human_scores.json",
        "real_kp_dir": "SAVE_REAL_ONLY_10_minus1",
        "gen_kp_dir": "generated_kps",
        "real_clip_dir": None,
        "real_dino_dir": None,
        "gen_clip_dir": None,
        "gen_dino_dir": None,
    },
    "modality_dims": {
        "raw": {
            "vit": 1024,
            "global": 9,
            "pose": 207,
            "beta": 10,
            "kp2d": 120,
            "clip": 512,
            "dino": 768,
        },
        "diff": {
            "vit": 1024,
            "global": 3,
            "pose": 69,
            "beta": 10,
            "kp2d": 120,
            "clip": 512,
            "dino": 768,
        },
    },
    "loss_weights": {
        "hard_negative": 10.0,
    },
    "eps": 1e-6,
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import HumanActionScorer
from losses import *
from utils import * 

SEED = GLOBAL_CONFIG["seed"]
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

g = torch.Generator()
g.manual_seed(SEED)

def _maybe_dataparallel(mod: nn.Module, use_dp: bool) -> nn.Module:
    if use_dp and torch.cuda.device_count() > 1:
        return nn.DataParallel(mod)
    return mod

def _save_state_dict(model: nn.Module, path: Path):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(sd, path)

def _instantiate_model_from_spec(spec, *, latent_dim: int, dims_map_raw: dict, dims_map_diff: dict, model_kwargs: dict) -> nn.Module:
    if isinstance(spec, nn.Module):
        return spec
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec(dims_map_raw=dims_map_raw, dims_map_diff=dims_map_diff, latent_dim=latent_dim, **(model_kwargs or {}))
    if callable(spec):
        return spec(dims_map_raw=dims_map_raw, dims_map_diff=dims_map_diff, latent_dim=latent_dim, **(model_kwargs or {}))
    raise ValueError(f"Unsupported model spec type: {type(spec)}")


def scan_gen_flat(root_dir: str):
    """Recursively list all .npz mesh files under root_dir."""
    meshes = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".npz"):
                meshes.append(os.path.join(dirpath, fn))
    return sorted(meshes)


class BaseExperiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.use_dp = False

        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = cfg.get("name") or self.__class__.__name__
        self.exp_name = f"{self.exp_name}_{timestamp}"
        self.save_dir = Path("SAVE")/self.exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if cfg["filter_classes"] is None:
            print("[Data] Using ALL classes for training.")
            self.full_ds = NpzVideoDataset(cfg["dataset_dir"], filter_classes=cfg.get("filter_classes", None))
            self.centroid_ds = NpzVideoDataset(cfg["dataset_dir"], filter_classes=["JumpingJack","PullUps","PushUps","HulaHoop","WallPushups","Shotput","SoccerJuggling","TennisSwing","ThrowDiscus","BodyWeightSquats"])
        else:
            self.full_ds = NpzVideoDataset(cfg["dataset_dir"], filter_classes=cfg.get("filter_classes", None))
            self.centroid_ds = NpzVideoDataset(cfg["dataset_dir"], filter_classes=["JumpingJack","PullUps","PushUps","HulaHoop","WallPushups","Shotput","SoccerJuggling","TennisSwing","ThrowDiscus","BodyWeightSquats"])
        self.train_ds, self.test_ds = train_test_split(self.full_ds, train_ratio=0.8, seed=SEED)

        self.stats = compute_stats_from_npz(self.train_ds.items, keypoint_dir=cfg["real_kp"], clip_dir=cfg.get("real_clip"), dino_dir=cfg.get("real_dino"))
        self.ALL_CLASSES = sorted({it.cls for it in self.full_ds.items})
        self.label_dict = {cls: i for i, cls in enumerate(self.ALL_CLASSES)}
        with open(self.save_dir/"label_mapping.json","w") as f:
            json.dump(self.label_dict,f,indent=2)

        probe = WindowDataset(
            sample_all_windows_npz(self.train_ds, clip_len=cfg["clip_len"], stride=cfg["stride"]),
            clip_len=cfg["clip_len"], keypoint_dir=cfg["real_kp"], stats=self.stats, seed=SEED,
            clip_dir=cfg.get("real_clip"),
            dino_dir=cfg.get("real_dino"),
        )[0]
        self.input_dim = probe[0].shape[-1]

        available_modalities = self._detect_available_modalities(cfg)
        print(f"[Modalities] Available: {', '.join(available_modalities)}")

        model_kwargs = cfg.get("model_kwargs", {})
        dims_map_raw, dims_map_diff = self._build_dims_map(available_modalities, cfg)
        print(f"[Modalities] dims_map_raw: {dims_map_raw}")
        print(f"[Modalities] dims_map_diff: {dims_map_diff}")
        
        if hasattr(self, "model") and self.model is not None:
            self.model = _instantiate_model_from_spec(
                self.model, latent_dim=cfg["latent_dim"], dims_map_raw=dims_map_raw, dims_map_diff=dims_map_diff, model_kwargs=model_kwargs
            )
        else:
            self.model = HumanActionScorer(
                dims_map_raw=dims_map_raw,
                dims_map_diff=dims_map_diff,
                latent_dim=cfg["latent_dim"],
                **model_kwargs
            )
        self.model = self.model.to(self.device)
        self.ce = nn.CrossEntropyLoss().to(self.device)

        params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=cfg["lr"])

        train_samples = sample_all_windows_npz(self.train_ds, clip_len=cfg["clip_len"], stride=cfg["stride"])
        self.train_window_ds = WindowDataset(train_samples, clip_len=cfg["clip_len"], stats=self.stats, keypoint_dir=cfg["real_kp"], seed=SEED, clip_dir=cfg.get("real_clip"), dino_dir=cfg.get("real_dino"))
        labels_for_sampler = [self.label_dict[it.cls] for (it, _s) in self.train_window_ds.samples]

        self.pk_sampler = PKBatchSampler(labels_for_sampler, P=cfg["P"], K=cfg["K"], drop_last=True)

        num_workers = int(cfg.get("num_workers", 0))
        self.train_loader = DataLoader(
            self.train_window_ds,
            batch_sampler=self.pk_sampler,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=safe_collate,
            pin_memory=(self.device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=steps_per_epoch * cfg["epochs"], eta_min=1e-6
        )

        self.best_eval_loss = float("inf")  # Lower is better (loss)

        self.human_corr_meshes = scan_gen_flat(GLOBAL_CONFIG["paths"]["mesh_human_data_generated"])

        # Compute centroids from training data (not a separate dataset) for consistency
        self.small_loader = make_test_loader(
            self.train_ds,
            clip_len=self.cfg["clip_len"],
            stride=self.cfg["stride"],
            stats=self.stats,
            seed=SEED,
            batch_size=2048,
            keypoint_dir=self.cfg["real_kp"],
            clip_dir=self.cfg.get("real_clip"),
            dino_dir=self.cfg.get("real_dino"),
            num_workers=int(self.cfg.get("num_workers", 0)),
        )
        
        # Create eval loader for test set evaluation
        self.eval_loader = make_test_loader(
            self.test_ds,
            clip_len=self.cfg["clip_len"],
            stride=self.cfg["stride"],
            stats=self.stats,
            seed=SEED,
            batch_size=2048,
            keypoint_dir=self.cfg["real_kp"],
            clip_dir=self.cfg.get("real_clip"),
            dino_dir=self.cfg.get("real_dino"),
            num_workers=int(self.cfg.get("num_workers", 0)),
        )

        print(f"Experiment '{self.exp_name}' initialized.")
        print(f"  Training on {len(self.train_ds)} videos ({len(self.train_window_ds)} windows)")
        print(f"  Evaluating on {len(self.test_ds)} videos")
        print(f"  Classes: {self.ALL_CLASSES}")

    def _detect_available_modalities(self, cfg: dict) -> list:
        """Detect which modalities are actually available based on probe sample."""
        modalities = ["vit", "global", "pose", "beta"]
        
        if cfg.get("real_kp") is not None:
            modalities.append("kp2d")
        if cfg.get("real_clip") is not None:
            modalities.append("clip")
        if cfg.get("real_dino") is not None:
            modalities.append("dino")
        
        return modalities

    def _build_dims_map(self, available_modalities: list, cfg: dict) -> tuple:
        """Build dims_map based on available modalities."""
        base_dims_raw = GLOBAL_CONFIG["modality_dims"]["raw"]
        base_dims_diff = GLOBAL_CONFIG["modality_dims"]["diff"]
        
        dims_map_raw = cfg.get("dims_map_raw", {})
        dims_map_diff = cfg.get("dims_map_diff", {})
        
        dims_map_raw_final = {}
        dims_map_diff_final = {}
        
        for mod in available_modalities:
            dims_map_raw_final[mod] = dims_map_raw.get(mod, base_dims_raw.get(mod, 0))
            dims_map_diff_final[mod] = dims_map_diff.get(mod, base_dims_diff.get(mod, 0))
        
        return dims_map_raw_final, dims_map_diff_final

    def compute_loss_components(self, emb, labels, feats=None) -> dict:
        raise NotImplementedError

    def train_one_epoch(self, epoch: int):
        cfg = self.cfg
        total_loss = 0.0

        for feats, cls_names, vids in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
            labels = torch.as_tensor([self.label_dict[c] for c in cls_names], dtype=torch.long).to(self.device)
            feats = feats.to(self.device)
            emb, _, _ = self.model(feats)

            loss_dict = self.compute_loss_components(emb, labels, feats)
            loss_val = sum(loss_dict.values())
            if not torch.isfinite(loss_val):
                continue

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss_val.item())

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self, epoch: int):
        """Build centroids from training data."""
        centroids_sub, _ = build_train_centroids_subset(self.model, self.small_loader, self.label_dict, device=self.device)
        return centroids_sub

    def evaluate_test_set(self, epoch: int):
        """
        Evaluate on test set: compute the same loss as training.
        Lower loss = better model.
        Returns: average loss and loss components
        """
        self.model.eval()
        device = self.device
        total_loss = 0.0
        loss_components_sum = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating test set"):
                if batch is None:
                    continue
                feats, cls_names, _ = batch
                feats = feats.to(device)
                labels = torch.as_tensor([self.label_dict[c] for c in cls_names], dtype=torch.long, device=device)
                
                emb, _, _ = self.model(feats)
                
                # Compute loss components (same as training)
                loss_dict = self.compute_loss_components(emb, labels, feats)
                loss_val = sum(loss_dict.values())
                
                if not torch.isfinite(loss_val):
                    continue
                
                total_loss += float(loss_val.item())
                
                # Accumulate loss components for logging
                for key, val in loss_dict.items():
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0.0
                    loss_components_sum[key] += float(val.item())
                
                num_batches += 1
        
        self.model.train()
        
        if num_batches == 0:
            return float("inf"), {}
        
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components_sum.items()}
        
        return avg_loss, avg_loss_components
    
    def evaluate_test_set_centroid_distance(self, epoch: int, centroids_sub):
        """
        Evaluate on test set: compute average distance to class centroids (for logging only).
        Returns: average centroid distance and per-class distances
        """
        self.model.eval()
        device = self.device
        
        # Collect embeddings and labels from test set
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.eval_loader:
                if batch is None:
                    continue
                feats, cls_names, _ = batch
                feats = feats.to(device)
                emb, _, _ = self.model(feats)
                
                labels = torch.as_tensor([self.label_dict[c] for c in cls_names], dtype=torch.long, device=device)
                all_embeddings.append(emb)
                all_labels.append(labels)
        
        if not all_embeddings:
            return float("inf"), {}
        
        # Concatenate all embeddings and labels
        all_emb = torch.cat(all_embeddings, dim=0)  # [N, dim]
        all_lbl = torch.cat(all_labels, dim=0)  # [N]
        
        # Normalize embeddings
        all_emb = F.normalize(all_emb, p=2, dim=-1)
        
        # Compute distances to centroids for each sample
        distances = []
        for i in range(len(all_emb)):
            label = all_lbl[i].item()
            if label >= len(centroids_sub):
                continue
            centroid = centroids_sub[label].to(device)
            emb = all_emb[i]
            dist = torch.norm(emb - centroid, p=2).item()
            distances.append(dist)
        
        if not distances:
            return float("inf"), {}
        
        avg_distance = float(np.mean(distances))
        
        # Also compute per-class distances for logging
        class_distances = {}
        for cls_name, cls_idx in self.label_dict.items():
            if cls_idx >= len(centroids_sub):
                continue
            mask = (all_lbl == cls_idx)
            if mask.sum() == 0:
                continue
            cls_emb = all_emb[mask]
            centroid = centroids_sub[cls_idx].to(device)
            cls_dists = torch.norm(cls_emb - centroid.unsqueeze(0), p=2, dim=-1)
            class_distances[cls_name] = float(cls_dists.mean().item())
        
        self.model.train()
        return avg_distance, class_distances

    def evaluate_human_corr(self, epoch: int, centroids_sub=None):
        """Evaluate human correlation (for logging only, not used for checkpointing)."""
        out_appearance, out_action, out_anatomy, out_motion = get_human_corr(
            self.human_corr_meshes,
            GLOBAL_CONFIG["paths"]["human_scores"],
            centroids_sub,
            self.label_dict,
            self.stats,
            self.model,
            clip_len=self.cfg["clip_len"],
            stride=self.cfg["stride"],
            gen_kp_dir=GLOBAL_CONFIG["paths"]["gen_kp_dir"],
            gen_clip_dir=GLOBAL_CONFIG["paths"]["gen_clip_dir"],
            gen_dino_dir=GLOBAL_CONFIG["paths"]["gen_dino_dir"],
        )

        all_dims = {
            "action": out_action,
            "motion": out_motion,
        }

        pretty_parts = []
        for dim_name, metrics in all_dims.items():
            for metric_name, val in metrics.items():
                val = float(val) if val is not None else float("nan")
                if val == val:
                    pretty_parts.append(f"{dim_name}.{metric_name}: {val:.4f}")
                else:
                    pretty_parts.append(f"{dim_name}.{metric_name}: NaN")

        print("[Eval] Human corr -> " + " | ".join(pretty_parts))

    def run(self):
        for epoch in range(self.cfg["epochs"]):
            avg_loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch+1} train loss {avg_loss:.4f}")
            
            # Build centroids from training data
            centroids_sub = self.evaluate(epoch)
            
            # Evaluate on test set using loss (for checkpointing)
            eval_loss, loss_components = self.evaluate_test_set(epoch)
            print(f"[Eval] Test set loss: {eval_loss:.4f}")
            
            # Log loss components
            if loss_components:
                component_parts = [f"{k}: {v:.4f}" for k, v in sorted(loss_components.items())]
                print(f"[Eval] Loss components: " + " | ".join(component_parts))
            
            # Checkpoint based on test set loss (lower is better)
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                ckpt = self.save_dir / f"best_eval_epoch{epoch+1:03d}_loss{eval_loss:.4f}.pt"
                _save_state_dict(self.model, ckpt)
                print(f"✅ Saved best eval checkpoint (loss: {eval_loss:.4f}) to {ckpt}")
            
            # Log centroid distances (for monitoring only)
            avg_distance, class_distances = self.evaluate_test_set_centroid_distance(epoch, centroids_sub)
            print(f"[Eval] Test set avg centroid distance: {avg_distance:.4f}")
            if class_distances:
                class_parts = [f"{cls}: {dist:.4f}" for cls, dist in sorted(class_distances.items())]
                print(f"[Eval] Per-class distances: " + " | ".join(class_parts))
            
            # Log human correlation (for monitoring, not checkpointing)
            self.evaluate_human_corr(epoch, centroids_sub)
        
        print("✅ Training complete.")

class Exp_TCL_Hard_V2Plus(BaseExperiment):
    def __init__(self):
        cfg = dict(
            dataset_dir=GLOBAL_CONFIG["paths"]["mesh_human_data_real"],
            batch_size=2048, latent_dim=128, epochs=30, clip_len=32, stride=8, P=10, K=24,
            total_windows_per_epoch=16384, windows_per_video=8, lr=3e-4, device="cuda",
            name="HumanActionScorer",
            filter_classes=["JumpingJack", "PullUps", "PushUps", "HulaHoop", "WallPushups",
                            "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "BodyWeightSquats"],
            model_kwargs={}, data_parallel=GLOBAL_CONFIG["use_dataparallel"],
            real_kp=GLOBAL_CONFIG["paths"]["real_kp_dir"],
        )
        self.model = HumanActionScorer
        super().__init__(cfg)
        self.tcl  = TCL().to(self.device)
        self.hard = SupConWithHardNegatives().to(self.device)

    

    def train_one_epoch(self, epoch: int):
        total_loss = 0.0

        for feats, cls_names, vids in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
            labels = torch.as_tensor([self.label_dict[c] for c in cls_names],
                                     dtype=torch.long, device=self.device)
            feats = feats.to(self.device)
            emb, frame_emb, _ = self.model(feats)

            loss_dict = self.compute_loss_components(emb, labels, feats)
            loss_val = sum(loss_dict.values())
            if not torch.isfinite(loss_val):
                continue

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss_val.item())

        return total_loss / max(1, len(self.train_loader))

    def compute_loss_components(self, emb, labels, feats=None):
        sh_emb, _, _ = self.model(partial_shuffle_within_window(feats))
        rev_emb, _, _ = self.model(reverse_sequence(feats))
        st_emb,  _, _ = self.model(get_static_window(feats))

        hard_weight = GLOBAL_CONFIG["loss_weights"]["hard_negative"]
        losses = {
            "tcl": self.tcl(emb, labels),
            "hard_shuf": hard_weight * self.hard(emb, emb, sh_emb),
            "hard_rev":  hard_weight * self.hard(emb, emb, rev_emb),
            "hard_stat": hard_weight * self.hard(emb, emb, st_emb),
        }

        return losses

if __name__ == "__main__":
    exp = Exp_TCL_Hard_V2Plus()
    exp.run()
