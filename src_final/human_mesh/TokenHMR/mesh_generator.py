from .tokenhmr.lib.models import load_tokenhmr
from .tokenhmr.lib.utils import recursive_to
from .tokenhmr.lib.utils.render_openpose import render_openpose
from .tokenhmr.lib.datasets.vitdet_dataset import (
    ViTDetDataset,
    DEFAULT_MEAN,
    DEFAULT_STD,
)
from .tokenhmr.lib.utils.renderer import Renderer, cam_crop_to_full
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import detectron2.data.transforms as T
# Load detector
from detectron2.engine.defaults import DefaultPredictor
from .tokenhmr.lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import lib
from pathlib import Path
import torch
import os
import cv2
import numpy as np
import tqdm
import json
import natsort
from diffusers.utils import export_to_video
import imageio
import pickle
import time
from torchvision.ops import nms
os.environ["PYOPENGL_PLATFORM"] = "egl"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from torch.utils.data import Dataset, DataLoader

class _FramesDS(Dataset):
    def __init__(self, frames): self.frames = frames
    def __len__(self): return len(self.frames)
    def __getitem__(self, i): return self.frames[i]  # np.ndarray(H,W,3)

def _collate_keep_list(batch):
    # DataLoader default would stack; we want a list of np arrays
    return batch  # list[np.ndarray]

def capture_dict_info(d, name, indent=4):
    """
    Recursively capture full tensor values instead of just shapes.
    Converts tensors to lists for JSON saving.
    """
    captured = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # Convert tensor to a Python list
            captured[k] = v.detach().cpu().tolist()
        elif isinstance(v, dict):
            captured[k] = capture_dict_info(v, name=k, indent=indent + 4)
        elif isinstance(v, list):
            captured[k] = [capture_dict_info(item, name=k, indent=indent + 4) if isinstance(item, dict) else item for item in v]
        else:
            captured[k] = v  # Save as-is if it's a number, string, etc.
    return captured
    
def create_video_from_masks(mask_folder, output_video_path, fps=8):
    mask_folder = Path(mask_folder)
    mask_files = list(mask_folder.glob('*_all_mask.png'))
    mask_files = natsort.natsorted(mask_files)

    if len(mask_files) == 0:
        raise ValueError(f"No mask files found in {mask_folder}")

    frames = []
    for mask_path in mask_files:
        mask = imageio.imread(mask_path)  # grayscale 0-255
        if mask.ndim == 2:  # ensure (H, W, 3)
            mask = np.stack([mask] * 3, axis=-1)

        mask = mask.astype(np.float32) / 255.0  # ⚡ normalize to [0, 1] range
        frames.append(mask)

    # Now frames are float32 images in [0,1] as expected
    export_to_video(frames, output_video_path, fps=fps)


class TokenHMRMeshGenerator:
    def __init__(self, config, device="cuda"):
        self.device = device
        self._config = {
            "checkpoint": "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
            "model_config": "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        }
        # merge with user config
        self.config = config
        self.config.update(self._config)
        model, model_cfg = load_tokenhmr(
            checkpoint_path=self.config["checkpoint"],
            model_cfg=self.config["model_config"],
            is_train_state=False,
            is_demo=True,
        )

        # Setup model
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)
        model.eval()
        self.model = model

        cfg_path = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/human_mesh/TokenHMR/tokenhmr/lib/configs/cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.detector = detector

        # Setup the renderer
        renderer = Renderer(model_cfg, faces=model.smpl.faces)
        self.renderer = renderer
        self.model_cfg = model_cfg


        self.det2_cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.det2_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.det2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.det2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.det2_predictor = DefaultPredictor(self.det2_cfg)

        self.det3 = instantiate(detectron2_cfg.model)
        self.det3.to("cuda")
        self.det3.eval()
        DetectionCheckpointer(self.det3).load(detectron2_cfg.train.init_checkpoint)

    def filter_single_person(self, frames):

        # metadata_path = os.path.join(input_folder_path, "metadata.json")
        # with open(metadata_path, "r") as f:
        #     metadata = json.load(f)
        # fps = metadata["fps"] if "fps" in metadata else 24

        # Iterate over all images in folder (do not meshify if more than 1 person detected)
        for img_cv2 in frames:

            # Detect humans in image
            det_out = self.det2_predictor(img_cv2)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            # print(len(boxes))

            if len(boxes) != 1:
                # print(f"Skipping {input_folder_path} at {img_path} because it has {len(boxes)} detections (expected 1)")
                return False
        return True
    def filter_single_person_batched(self, frames, batch_size: int = 16, score_thr: float = 0.5) -> bool:
        """
        Returns True iff every (sampled) frame has exactly one 'person' detection with score > score_thr.
        Early-exits on first failure.
        """
        print(len(frames))
        exit()
        if len(frames) == 0:
            return False

        for chunk in _to_batches(frames, batch_size):
            # many detectors accept list[np.ndarray]; if yours needs tensors, convert here
            det_out = self.detector(chunk)  # expected: iterable of dicts with "instances"
            print(det_out)
            # normalize to list
            if isinstance(det_out, dict) and "instances" in det_out:
                det_out = [det_out]
            for out in det_out:
                inst = out["instances"]
                valid_idx = (inst.pred_classes == 0) & (inst.scores > score_thr)
                # Count persons in THIS frame:
                num = int(valid_idx.sum().item())
                if num != 1:
                    return False
        return True

    def generate_mesh_from_frames(self, frames):

        
        fps = 25


        # Make list to collect masked real frames
        masked_real_mesh_frames = []
        masked_real_bbox_frames = []
        masked_real_hybrid_frames =[]
        mesh_frames = []  # Add list to collect mesh renderings
        overlay_frames = []  # Add list to collect overlay frames

        out_dir = "DEL"

        

        os.makedirs(out_dir, exist_ok=True)

        # Iterate over all images in folder
        for idx, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            img_cv2 = frame

            # Detect humans in image
            det_out = self.det2_predictor(img_cv2)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            # det_instances = det_out["instances"]
            # person_mask = (det_instances.pred_classes == 0)
            # score_mask  = (det_instances.scores > 0.5)
            # mask = person_mask & score_mask

            # boxes_t  = det_instances.pred_boxes.tensor[mask]      # [N,4]
            # scores_t = det_instances.scores[mask]                 # [N]

            # # Hard-NMS to kill near-duplicates (tune IoU as needed)
            # keep_idx = nms(boxes_t, scores_t, iou_threshold=0.5)  # try 0.4–0.6
            # boxes_t  = boxes_t[keep_idx]
            # scores_t = scores_t[keep_idx]
            # boxes = boxes_t.cpu().numpy()

            # Skip frames without detections
            if len(boxes) == 0:
                continue

            # Run on all detected humans
            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=False, num_workers=0
            )

            all_verts = []
            all_cam_t = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self.model(batch)

                pred_cam = out["pred_cam"]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = (
                    self.model_cfg.EXTRA.FOCAL_LENGTH
                    / self.model_cfg.MODEL.IMAGE_SIZE
                    * img_size.max()
                )
                pred_cam_t_full = (
                    cam_crop_to_full(
                        pred_cam, box_center, box_size, img_size, scaled_focal_length
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    person_id = int(batch["personid"][n])
                    white_img = (
                        torch.ones_like(batch["img"][n]).cpu()
                        - DEFAULT_MEAN[:, None, None] / 255
                    ) / (DEFAULT_STD[:, None, None] / 255)
                    input_patch = batch["img"][n].cpu() * (
                        DEFAULT_STD[:, None, None] / 255
                    ) + (DEFAULT_MEAN[:, None, None] / 255)
                    input_patch = input_patch.permute(1, 2, 0).numpy()

                    regression_img = self.renderer(
                        out["pred_vertices"][n].detach().cpu().numpy(),
                        out["pred_cam_t"][n].detach().cpu().numpy(),
                        batch["img"][n],
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                    )

                    mask_img_rgba = self.renderer.render_rgba(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        cam_t=out['pred_cam_t'][n].detach().cpu().numpy(),
                        render_res=[batch['img'][n].shape[-1], batch['img'][n].shape[-1]]
                    )

                    mask_alpha = mask_img_rgba[:, :, 3]
                    body_mask = (mask_alpha > 0).astype(np.uint8)

                    # mask_save_path = os.path.join(out_dir, f'{img_fn}_{person_id}_mask.png')
                    # cv2.imwrite(mask_save_path, body_mask * 255)

                    if self.config["side_view"]:
                        side_img = self.renderer(
                            out["pred_vertices"][n].detach().cpu().numpy(),
                            out["pred_cam_t"][n].detach().cpu().numpy(),
                            white_img,
                            mesh_base_color=LIGHT_BLUE,
                            scene_bg_color=(1, 1, 1),
                            side_view=True,
                        )
                        final_img = np.concatenate(
                            [input_patch, regression_img, side_img], axis=1
                        )
                    else:
                        final_img = np.concatenate(
                            [input_patch, regression_img], axis=1
                        )

                    out_img_path = os.path.join(out_dir, f"{idx}_{person_id}.png")
                    cv2.imwrite(
                        out_img_path,
                        255 * final_img[:, :, ::-1],
                    )

                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                    # if self.config["save_mesh"]:
                    #     camera_translation = cam_t.copy()
                    #     tmesh = self.renderer.vertices_to_trimesh(
                    #         verts, camera_translation, LIGHT_BLUE
                    #     )
                    #     mesh_out_path = os.path.join(
                    #         out_dir, f"{img_fn}_{person_id}.obj"
                    #     )
                    #     tmesh.export(mesh_out_path)

            # -------- Full frame render --------
            if self.config["full_frame"] and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = self.renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args
                )

                # Extract mesh body alpha
                mask_alpha_all = cam_view[:, :, 3]   # (H,W)
                body_mask_all = (mask_alpha_all > 0).astype(np.uint8)  # binary 0/1

                # Prepare grayscale input frame
                gray_real_frame = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

                # --- 1. Mesh Alpha Masked Frame ---
                masked_real_mesh = gray_real_frame * body_mask_all

                # --- 2. BBox Masked Frame ---
                h, w = gray_real_frame.shape[:2]
                bbox_mask = np.zeros((h, w), dtype=np.uint8)

                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = np.clip(x1, 0, w-1)
                    x2 = np.clip(x2, 0, w-1)
                    y1 = np.clip(y1, 0, h-1)
                    y2 = np.clip(y2, 0, h-1)
                    bbox_mask[y1:y2, x1:x2] = 1

                masked_real_bbox = gray_real_frame * bbox_mask

                # --- 3. Hybrid Mask: bbox cropped + alpha inside bbox ---
                hybrid_mask = np.zeros_like(gray_real_frame, dtype=np.uint8)

                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = np.clip(x1, 0, w-1)
                    x2 = np.clip(x2, 0, w-1)
                    y1 = np.clip(y1, 0, h-1)
                    y2 = np.clip(y2, 0, h-1)

                    if x2 <= x1 or y2 <= y1:
                        continue  # Skip invalid boxes

                    region_real = gray_real_frame[y1:y2, x1:x2]
                    region_alpha = body_mask_all[y1:y2, x1:x2]

                    # Ensure binary mask (0 or 1)
                    region_alpha = (region_alpha > 0.2).astype(np.uint8) # FIXME

                    # Apply mask inside bbox
                    refined = region_real * region_alpha

                    # Place into hybrid mask
                    hybrid_mask[y1:y2, x1:x2] = refined

                
                # --- Save each for motion maps later ---
                masked_real_mesh_frames.append(masked_real_mesh)
                masked_real_bbox_frames.append(masked_real_bbox)
                masked_real_hybrid_frames.append(hybrid_mask)

                # --- Save Overlay visualization ---
                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0  # Convert BGR to RGB
                input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
                input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                # cv2.imwrite(os.path.join(out_dir, f'{img_fn}_all.png'), (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8))
                
                # Save mesh frame for video
                mesh_frame = (255 * cam_view[:, :, :3]).astype(np.uint8)
                mesh_frames.append(mesh_frame)
                
                # Save overlay frame for video - keep in RGB format
                overlay_frame = (255 * input_img_overlay).astype(np.uint8)
                overlay_frames.append(overlay_frame)

        # # After full loop
        # print(len(masked_real_bbox_frames))
        # exit()
        if len(masked_real_bbox_frames) >= 2:
            print("\n Computing motion from masked real frames...")

            def compute_motion(frames, threshold=10):
                motion_frames = []
                for i in range(1, len(frames)):
                    diff = cv2.absdiff(frames[i], frames[i-1])
                    _, binary_motion = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                    motion_frames.append(binary_motion)
                blank = np.zeros_like(motion_frames[0])
                motion_frames = [blank] + motion_frames
                motion_frames_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in motion_frames]
                return motion_frames_rgb

            # # === Compute motions separately ===
            # motion_mesh_rgb = compute_motion(masked_real_mesh_frames)
            # motion_bbox_rgb = compute_motion(masked_real_bbox_frames)
            # motion_hybrid_rgb = compute_motion(masked_real_hybrid_frames)

            # # === Save motion videos ===
            # export_to_video(motion_mesh_rgb, os.path.join(out_dir, "motion_masked_mesh.mp4"), fps=fps)
            # export_to_video(motion_bbox_rgb, os.path.join(out_dir, "motion_masked_bbox.mp4"), fps=fps)
            # export_to_video(motion_hybrid_rgb, os.path.join(out_dir, "motion_masked_hybrid.mp4"), fps=fps)

            # === Save mesh videos ===
            if len(mesh_frames) > 0:
                print("\n Creating mesh videos...")
                # # Save mesh-only video
                # mesh_frames_rgb = [(f.astype(np.float32) / 255.0) for f in mesh_frames]
                # export_to_video(mesh_frames_rgb, os.path.join(out_dir, "mesh_renderings.mp4"), fps=fps)
                
                # Save overlay video (mesh on original frames)
                overlay_frames_rgb = [(f.astype(np.float32) / 255.0) for f in overlay_frames]
                export_to_video(overlay_frames_rgb, os.path.join(out_dir, "mesh_overlay.mp4"), fps=fps)

    # def generate_mesh_from_frames(self, input_folder_path, out_dir):

    #     metadata_path = os.path.join(input_folder_path, "metadata.json")
    #     with open(metadata_path, "r") as f:
    #         metadata = json.load(f)
    #     fps = metadata["fps"] if "fps" in metadata else 24


    #     # Make list to collect masked real frames
    #     masked_real_mesh_frames = []
    #     masked_real_bbox_frames = []
    #     masked_real_hybrid_frames =[]
    #     mesh_frames = []  # Add list to collect mesh renderings
    #     overlay_frames = []  # Add list to collect overlay frames

    #     # Iterate over all images in folder (do not meshify if more than 1 person detected)
    #     for img_path in tqdm.tqdm(sorted(list(Path(input_folder_path).glob("*.jpg")) + list(Path(input_folder_path).glob("*.png")))):
    #         print(f"Processing {img_path}..")
    #         img_cv2 = cv2.imread(str(img_path))

    #         # Detect humans in image
    #         det_out = self.detector(img_cv2)
    #         det_instances = det_out["instances"]
    #         valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    #         boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    #         if len(boxes) != 1:
    #             print(f"Skipping {input_folder_path} at {img_path} because it has {len(boxes)} detections (expected 1)")
    #             return

    #     os.makedirs(out_dir, exist_ok=True)

    #     # Iterate over all images in folder
    #     for img_path in tqdm.tqdm(sorted(list(Path(input_folder_path).glob("*.jpg")) + list(Path(input_folder_path).glob("*.png")))):
    #         print(f"Processing {img_path}..")
    #         img_cv2 = cv2.imread(str(img_path))

    #         # Detect humans in image
    #         det_out = self.detector(img_cv2)
    #         det_instances = det_out["instances"]
    #         valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    #         boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    #         # Skip frames without detections
    #         if len(boxes) == 0:
    #             continue

    #         # Run on all detected humans
    #         dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes)
    #         dataloader = torch.utils.data.DataLoader(
    #             dataset, batch_size=8, shuffle=False, num_workers=0
    #         )

    #         all_verts = []
    #         all_cam_t = []

    #         for batch in dataloader:
    #             batch = recursive_to(batch, self.device)
    #             with torch.no_grad():
    #                 out = self.model(batch)

    #             dict_info = capture_dict_info(out, name="model_out")
    #             # save_json_path = os.path.join(out_dir, f'{img_path.stem}_info.json')
    #             # with open(save_json_path, 'w') as f:
    #             #     json.dump(dict_info, f, indent=4)

    #             # save also as .pkl
    #             save_pkl_path = os.path.join(out_dir, f'{img_path.stem}_info.pkl')
    #             with open(save_pkl_path, 'wb') as f:
    #                 pickle.dump(dict_info, f)

    #             pred_cam = out["pred_cam"]
    #             box_center = batch["box_center"].float()
    #             box_size = batch["box_size"].float()
    #             img_size = batch["img_size"].float()
    #             scaled_focal_length = (
    #                 self.model_cfg.EXTRA.FOCAL_LENGTH
    #                 / self.model_cfg.MODEL.IMAGE_SIZE
    #                 * img_size.max()
    #             )
    #             pred_cam_t_full = (
    #                 cam_crop_to_full(
    #                     pred_cam, box_center, box_size, img_size, scaled_focal_length
    #                 )
    #                 .detach()
    #                 .cpu()
    #                 .numpy()
    #             )

    #             batch_size = batch["img"].shape[0]
    #             for n in range(batch_size):
    #                 img_fn, _ = os.path.splitext(os.path.basename(img_path))
    #                 person_id = int(batch["personid"][n])
    #                 white_img = (
    #                     torch.ones_like(batch["img"][n]).cpu()
    #                     - DEFAULT_MEAN[:, None, None] / 255
    #                 ) / (DEFAULT_STD[:, None, None] / 255)
    #                 input_patch = batch["img"][n].cpu() * (
    #                     DEFAULT_STD[:, None, None] / 255
    #                 ) + (DEFAULT_MEAN[:, None, None] / 255)
    #                 input_patch = input_patch.permute(1, 2, 0).numpy()

    #                 regression_img = self.renderer(
    #                     out["pred_vertices"][n].detach().cpu().numpy(),
    #                     out["pred_cam_t"][n].detach().cpu().numpy(),
    #                     batch["img"][n],
    #                     mesh_base_color=LIGHT_BLUE,
    #                     scene_bg_color=(1, 1, 1),
    #                 )

    #                 mask_img_rgba = self.renderer.render_rgba(
    #                     out['pred_vertices'][n].detach().cpu().numpy(),
    #                     cam_t=out['pred_cam_t'][n].detach().cpu().numpy(),
    #                     render_res=[batch['img'][n].shape[-1], batch['img'][n].shape[-1]]
    #                 )

    #                 mask_alpha = mask_img_rgba[:, :, 3]
    #                 body_mask = (mask_alpha > 0).astype(np.uint8)

    #                 mask_save_path = os.path.join(out_dir, f'{img_fn}_{person_id}_mask.png')
    #                 cv2.imwrite(mask_save_path, body_mask * 255)

    #                 if self.config["side_view"]:
    #                     side_img = self.renderer(
    #                         out["pred_vertices"][n].detach().cpu().numpy(),
    #                         out["pred_cam_t"][n].detach().cpu().numpy(),
    #                         white_img,
    #                         mesh_base_color=LIGHT_BLUE,
    #                         scene_bg_color=(1, 1, 1),
    #                         side_view=True,
    #                     )
    #                     final_img = np.concatenate(
    #                         [input_patch, regression_img, side_img], axis=1
    #                     )
    #                 else:
    #                     final_img = np.concatenate(
    #                         [input_patch, regression_img], axis=1
    #                     )

    #                 out_img_path = os.path.join(out_dir, f"{img_fn}_{person_id}.png")
    #                 cv2.imwrite(
    #                     out_img_path,
    #                     255 * final_img[:, :, ::-1],
    #                 )

    #                 verts = out["pred_vertices"][n].detach().cpu().numpy()
    #                 cam_t = pred_cam_t_full[n]
    #                 all_verts.append(verts)
    #                 all_cam_t.append(cam_t)

    #                 # if self.config["save_mesh"]:
    #                 #     camera_translation = cam_t.copy()
    #                 #     tmesh = self.renderer.vertices_to_trimesh(
    #                 #         verts, camera_translation, LIGHT_BLUE
    #                 #     )
    #                 #     mesh_out_path = os.path.join(
    #                 #         out_dir, f"{img_fn}_{person_id}.obj"
    #                 #     )
    #                 #     tmesh.export(mesh_out_path)

    #         # -------- Full frame render --------
    #         if self.config["full_frame"] and len(all_verts) > 0:
    #             misc_args = dict(
    #                 mesh_base_color=LIGHT_BLUE,
    #                 scene_bg_color=(1, 1, 1),
    #                 focal_length=scaled_focal_length,
    #             )
    #             cam_view = self.renderer.render_rgba_multiple(
    #                 all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args
    #             )

    #             # Extract mesh body alpha
    #             mask_alpha_all = cam_view[:, :, 3]   # (H,W)
    #             body_mask_all = (mask_alpha_all > 0).astype(np.uint8)  # binary 0/1

    #             # Prepare grayscale input frame
    #             gray_real_frame = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    #             # --- 1. Mesh Alpha Masked Frame ---
    #             masked_real_mesh = gray_real_frame * body_mask_all

    #             # --- 2. BBox Masked Frame ---
    #             h, w = gray_real_frame.shape[:2]
    #             bbox_mask = np.zeros((h, w), dtype=np.uint8)

    #             for box in boxes:
    #                 x1, y1, x2, y2 = box.astype(int)
    #                 x1 = np.clip(x1, 0, w-1)
    #                 x2 = np.clip(x2, 0, w-1)
    #                 y1 = np.clip(y1, 0, h-1)
    #                 y2 = np.clip(y2, 0, h-1)
    #                 bbox_mask[y1:y2, x1:x2] = 1

    #             masked_real_bbox = gray_real_frame * bbox_mask

    #             # --- 3. Hybrid Mask: bbox cropped + alpha inside bbox ---
    #             hybrid_mask = np.zeros_like(gray_real_frame, dtype=np.uint8)

    #             for box in boxes:
    #                 x1, y1, x2, y2 = box.astype(int)
    #                 x1 = np.clip(x1, 0, w-1)
    #                 x2 = np.clip(x2, 0, w-1)
    #                 y1 = np.clip(y1, 0, h-1)
    #                 y2 = np.clip(y2, 0, h-1)

    #                 if x2 <= x1 or y2 <= y1:
    #                     continue  # Skip invalid boxes

    #                 region_real = gray_real_frame[y1:y2, x1:x2]
    #                 region_alpha = body_mask_all[y1:y2, x1:x2]

    #                 # Ensure binary mask (0 or 1)
    #                 region_alpha = (region_alpha > 0.2).astype(np.uint8) # FIXME

    #                 # Apply mask inside bbox
    #                 refined = region_real * region_alpha

    #                 # Place into hybrid mask
    #                 hybrid_mask[y1:y2, x1:x2] = refined

                
    #             # --- Save each for motion maps later ---
    #             masked_real_mesh_frames.append(masked_real_mesh)
    #             masked_real_bbox_frames.append(masked_real_bbox)
    #             masked_real_hybrid_frames.append(hybrid_mask)

    #             # --- Save Overlay visualization ---
    #             input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0  # Convert BGR to RGB
    #             input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    #             input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    #             # cv2.imwrite(os.path.join(out_dir, f'{img_fn}_all.png'), (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8))
                
    #             # Save mesh frame for video
    #             mesh_frame = (255 * cam_view[:, :, :3]).astype(np.uint8)
    #             mesh_frames.append(mesh_frame)
                
    #             # Save overlay frame for video - keep in RGB format
    #             overlay_frame = (255 * input_img_overlay).astype(np.uint8)
    #             overlay_frames.append(overlay_frame)

    #     # # After full loop
    #     # print(len(masked_real_bbox_frames))
    #     # exit()
    #     if len(masked_real_bbox_frames) >= 2:
    #         print("\n Computing motion from masked real frames...")

    #         def compute_motion(frames, threshold=10):
    #             motion_frames = []
    #             for i in range(1, len(frames)):
    #                 diff = cv2.absdiff(frames[i], frames[i-1])
    #                 _, binary_motion = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    #                 motion_frames.append(binary_motion)
    #             blank = np.zeros_like(motion_frames[0])
    #             motion_frames = [blank] + motion_frames
    #             motion_frames_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in motion_frames]
    #             return motion_frames_rgb

    #         # === Compute motions separately ===
    #         motion_mesh_rgb = compute_motion(masked_real_mesh_frames)
    #         motion_bbox_rgb = compute_motion(masked_real_bbox_frames)
    #         motion_hybrid_rgb = compute_motion(masked_real_hybrid_frames)

    #         # # === Save motion videos ===
    #         # export_to_video(motion_mesh_rgb, os.path.join(out_dir, "motion_masked_mesh.mp4"), fps=fps)
    #         # export_to_video(motion_bbox_rgb, os.path.join(out_dir, "motion_masked_bbox.mp4"), fps=fps)
    #         # export_to_video(motion_hybrid_rgb, os.path.join(out_dir, "motion_masked_hybrid.mp4"), fps=fps)

    #         # === Save mesh videos ===
    #         if len(mesh_frames) > 0:
    #             print("\n Creating mesh videos...")
    #             # # Save mesh-only video
    #             # mesh_frames_rgb = [(f.astype(np.float32) / 255.0) for f in mesh_frames]
    #             # export_to_video(mesh_frames_rgb, os.path.join(out_dir, "mesh_renderings.mp4"), fps=fps)
                
    #             # Save overlay video (mesh on original frames)
    #             overlay_frames_rgb = [(f.astype(np.float32) / 255.0) for f in overlay_frames]
    #             export_to_video(overlay_frames_rgb, os.path.join(out_dir, "mesh_overlay.mp4"), fps=fps)

    #     print("\n Done!")

    #     # # === Side-by-side-by-side video ===
    #     # print("\n Creating side-by-side-by-side videos...")

    #     # real_frames_paths = sorted(list(Path(input_folder_path).glob('*.png')) + list(Path(input_folder_path).glob('*.jpg')))
    #     # real_frames = []
    #     # for path in real_frames_paths:
    #     #     frame = imageio.imread(path)
    #     #     if frame.ndim == 2:
    #     #         frame = np.stack([frame] * 3, axis=-1)
    #     #     frame = frame.astype(np.float32) / 255.0
    #     #     real_frames.append(frame)

    #     # masked_mesh_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in masked_real_mesh_frames]
    #     # masked_bbox_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in masked_real_bbox_frames]
    #     # masked_hybrid_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in masked_real_hybrid_frames]

    #     # # === Align all lengths ===
    #     # min_len = min(len(real_frames), len(masked_mesh_rgb), len(masked_bbox_rgb), len(masked_hybrid_rgb))
    #     # real_frames = real_frames[:min_len]
    #     # masked_mesh_rgb = masked_mesh_rgb[:min_len]
    #     # masked_bbox_rgb = masked_bbox_rgb[:min_len]
    #     # masked_hybrid_rgb = masked_hybrid_rgb[:min_len]

    #     # # === Save separate combined videos ===
    #     # def export_combined(real_frames, masked_frames, motion_frames, save_path):
    #     #     combined_frames = []
    #     #     for real_f, masked_f, motion_f in zip(real_frames, masked_frames, motion_frames):
    #     #         combined = np.concatenate([real_f, masked_f, motion_f], axis=1)
    #     #         combined_frames.append(combined)
    #     #     export_to_video(combined_frames, save_path, fps=fps)

    #     # export_combined(real_frames, masked_mesh_rgb, motion_mesh_rgb, os.path.join(
    #     #     out_dir, "combined_masked_mesh.mp4"
    #     # ))
    #     # export_combined(real_frames, masked_bbox_rgb, motion_bbox_rgb, os.path.join(
    #     #     out_dir, "combined_masked_bbox.mp4"
    #     # ))

    def process_video(self, frames):
        """
        Args:
            frames: list[np.ndarray(H,W,3)] or np.ndarray(T,H,W,3)
        Returns:
            mesh_info: dict {frame_idx: {pose, betas, global_orient, vit}}
        """
        mesh_info = {}

        # print GB sizes of detector and model
        det_params = sum(p.numel() for p in self.det2_predictor.model.parameters())
        model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Detector params: {det_params / 1e6:.1f}M")
        print(f"TokenHMR params: {model_params / 1e6:.1f}M")

        # ---- 1. Run detector on all frames at once ----
        start_time = time.time()
        # print model size of detector
        det_batch_size = 8          # tune for your GPU
        det_num_workers = 0          # can set >0 if your augmentations are CPU-side
        det_loader = DataLoader(
            _FramesDS(frames),
            batch_size=det_batch_size,
            shuffle=False,           # keep original order
            num_workers=det_num_workers,
            collate_fn=_collate_keep_list,
            pin_memory=False
        )

        # start_time_det = time.time()
        # det_outs = []
        # with torch.no_grad():
        #     for imgs in det_loader:
        #         # imgs is List[np.ndarray]; your DefaultPredictor_Lazy.__call__ should handle lists
        #         outs = self.detector(imgs)
        #         print(outs)
        #         break
        #         # normalize to list[dict]
        #         if isinstance(outs, dict):
        #             outs = [outs]
        #         det_outs.extend(outs)
        # assert isinstance(det_outs, (list, tuple)), "detector must support list of images"
        # end_time_det = time.time()
        # print(f"Detector running time: {end_time_det - start_time_det:.2f} seconds for {len(frames)} frames")
        # end_time = time.time()
        # print(f"Detector processed {len(frames)} frames in {end_time - start_time:.2f} seconds")
        # # # exit()

        
        # cfg = get_cfg()
        # # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # predictor = DefaultPredictor(cfg)
        start_time_det = time.time()
        det_outs = []
        with torch.no_grad():
            for imgs in frames:
                # imgs is List[np.ndarray]; your DefaultPredictor_Lazy.__call__ should handle lists
                # imgs = imgs[:, :, ::-1]  # BGR to RGB
                outs = self.det2_predictor(imgs)
                # normalize to list[dict]
                if isinstance(outs, dict):
                    outs = [outs]
                det_outs.extend(outs)
        end_time_det = time.time()
        print(f"Detectron2 default model running time: {end_time_det - start_time_det:.2f} seconds for {len(frames)} frames")

        # with torch.no_grad():
        #     for img in frames:
        #         img = img[:, :, ::-1]
        #         img = T.ResizeShortestEdge(short_edge_length=800, max_size=1333).get_transform(img).apply_image(img)
        #         img_tensor = torch.as_tensor(img.astype("float32").transpose(2,0,1))
        #         outs = self.det3([{'image': img_tensor}])[0]
        #         if isinstance(outs, dict):
        #             outs = [outs]
        #         det_outs.extend(outs)
        # end_time_det = time.time()
        # print(f"Detectron2 default model running time: {end_time_det - start_time_det:.2f} seconds for {len(frames)} frames")


        start_time = time.time()
        all_boxes = []
        valid_frames = []
        for idx, det_out in enumerate(det_outs):
            det_instances = det_out["instances"]
            person_mask = (det_instances.pred_classes == 0)
            score_mask  = (det_instances.scores > 0.5)
            mask = person_mask & score_mask

            boxes_t  = det_instances.pred_boxes.tensor[mask]      # [N,4]
            scores_t = det_instances.scores[mask]                 # [N]

            # Hard-NMS to kill near-duplicates (tune IoU as needed)
            keep_idx = nms(boxes_t, scores_t, iou_threshold=0.5)  # try 0.4–0.6
            boxes_t  = boxes_t[keep_idx]
            scores_t = scores_t[keep_idx]

            if boxes_t.shape[0] != 1:
                # v = Visualizer(frames[idx][:, :, ::-1], MetadataCatalog.get(self.det2_cfg.DATASETS.TRAIN[0]), scale=1.2)
                # out = v.draw_instance_predictions(det_out["instances"].to("cpu"))
                # cv2.imwrite(f"vis_{idx:04d}.png", out.get_image()[:, :, ::-1])
                # print(f"Skipping frame {idx} because it has {len(boxes_t)} detections (expected 1)")
                # exit()
                continue
            all_boxes.append(boxes_t.cpu().numpy())
            valid_frames.append((idx, frames[idx]))

        if not valid_frames:
            return False

        # if over 20% frames are invalid, skip entire video
        if len(valid_frames) < 0.8 * len(frames):
            print(f"Skipping video because only {len(valid_frames)}/{len(frames)} frames are valid")
            return False

        # ---- 2. Build dataset over multiple frames ----
        imgs = [f for _, f in valid_frames]
        boxes_list = all_boxes
        dataset = ViTDetDataset(self.model_cfg, img_cv2=imgs, boxes=boxes_list)
        print(len(dataset), "valid frames with single-person detections")

        # ---- 3. Run through DataLoader once ----
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
            print(f"TokenHMR processed {len(dataset)} frames in {end_time - start_time:.2f} seconds")

        # ---- 4. Flatten and regroup per frame ----
        poses = torch.cat([o['pred_smpl_params']['body_pose']     for o in results], dim=0)  # [N, ...]
        betas = torch.cat([o['pred_smpl_params']['betas']         for o in results], dim=0)
        gori  = torch.cat([o['pred_smpl_params']['global_orient'] for o in results], dim=0)
        vit   = torch.cat([o['pred_smpl_params']['token_out']     for o in results], dim=0)

        # dataset.flat_index is list of (frame_idx_in_valid_frames, person_idx)
        for sample_idx, (t, _) in enumerate(dataset.flat_index):
            mesh_info[valid_frames[t][0]] = {
                "pose":          poses[sample_idx].detach().cpu().numpy(),
                "betas":         betas[sample_idx].detach().cpu().numpy(),
                "global_orient": gori[sample_idx].detach().cpu().numpy(),
                "vit":           vit[sample_idx].detach().cpu().numpy(),
            }

        return mesh_info
            
            

if __name__ == "__main__":
    print("TokenHMR Mesh Generator")
