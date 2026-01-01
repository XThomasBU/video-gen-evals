from .lib.models import load_tokenhmr
from .lib.utils import recursive_to
from .lib.datasets.vitdet_dataset import (
    ViTDetDataset,
    DEFAULT_MEAN,
    DEFAULT_STD,
)
from .lib.utils.renderer import Renderer, cam_crop_to_full
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import DefaultPredictor
from .lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from pathlib import Path
import torch
import os
import cv2
import numpy as np
import time
os.environ["PYOPENGL_PLATFORM"] = "egl"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg


class TokenHMRMeshGenerator:
    def __init__(self, config, device="cuda"):
        self.device = device
        
        base_dir = Path(__file__).parent.parent
        checkpoint_path = base_dir / "data" / "checkpoints" / "tokenhmr_model_latest.ckpt"
        model_config_path = base_dir / "data" / "checkpoints" / "model_config.yaml"
        cfg_path = base_dir / "tokenhmr" / "lib" / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        
        self._config = {
            "checkpoint": str(checkpoint_path),
            "model_config": str(model_config_path),
        }
        self.config = config
        self.config.update(self._config)
        
        model, model_cfg = load_tokenhmr(
            checkpoint_path=self.config["checkpoint"],
            model_cfg=self.config["model_config"],
            is_train_state=False,
            is_demo=True,
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()
        self.model = model

        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.detector = detector

        renderer = Renderer(model_cfg, faces=model.smpl.faces)
        self.renderer = renderer
        self.model_cfg = model_cfg

        self.det2_cfg = get_cfg()
        self.det2_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.det2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
        self.det2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.det2_predictor = DefaultPredictor(self.det2_cfg)

        self.det3 = instantiate(detectron2_cfg.model)
        self.det3.to("cuda")
        self.det3.eval()
        DetectionCheckpointer(self.det3).load(detectron2_cfg.train.init_checkpoint)

    def filter_single_person(self, frames):
        for img_cv2 in frames:
            det_out = self.det2_predictor(img_cv2)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

            if len(boxes) != 1:
                return False
        return True

    def process_video(self, frames):
        """
        Args:
            frames: list[np.ndarray(H,W,3)] or np.ndarray(T,H,W,3)
        Returns:
            mesh_info: dict {frame_idx: {pose, betas, global_orient, vit}}
        """
        mesh_info = {}

        start_time = time.time()
        all_boxes = []
        valid_frames = []
        for idx, f in enumerate(frames):
            det_out = self.det2_predictor(f)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            if len(boxes) != 1:
                continue
            all_boxes.append(boxes)
            valid_frames.append((idx, frames[idx]))

        if not valid_frames:
            return False

        if len(valid_frames) < 0.8 * len(frames):
            return False

        imgs = [f for _, f in valid_frames]
        boxes_list = all_boxes
        
        class MultiFrameDataset(torch.utils.data.Dataset):
            def __init__(self, cfg, imgs, boxes, valid_frames):
                self.cfg = cfg
                self.imgs = imgs
                self.valid_frames = valid_frames
                self.datasets = []
                self.flat_index = []
                
                for i, (img, box) in enumerate(zip(imgs, boxes)):
                    ds = ViTDetDataset(cfg, img_cv2=img, boxes=box)
                    self.datasets.append(ds)
                    for j in range(len(ds)):
                        self.flat_index.append((i, j))
            
            def __len__(self):
                return len(self.flat_index)
            
            def __getitem__(self, idx):
                frame_idx, box_idx = self.flat_index[idx]
                return self.datasets[frame_idx][box_idx]
        
        dataset = MultiFrameDataset(self.model_cfg, imgs, boxes_list, valid_frames)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )

        results = []
        with torch.no_grad():
            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                out = self.model(batch)
                results.append(out)
            end_time = time.time()

        poses = torch.cat([o['pred_smpl_params']['body_pose'] for o in results], dim=0)
        betas = torch.cat([o['pred_smpl_params']['betas'] for o in results], dim=0)
        gori = torch.cat([o['pred_smpl_params']['global_orient'] for o in results], dim=0)
        vit = torch.cat([o['pred_smpl_params']['token_out'] for o in results], dim=0)

        for sample_idx, (frame_idx, _) in enumerate(dataset.flat_index):
            original_frame_idx = valid_frames[frame_idx][0]
            mesh_info[original_frame_idx] = {
                "pose": poses[sample_idx].detach().cpu().numpy(),
                "betas": betas[sample_idx].detach().cpu().numpy(),
                "global_orient": gori[sample_idx].detach().cpu().numpy(),
                "vit": vit[sample_idx].detach().cpu().numpy(),
            }

        return mesh_info


if __name__ == "__main__":
    print("TokenHMR Mesh Generator")
