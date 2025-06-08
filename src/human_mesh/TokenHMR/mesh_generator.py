from .tokenhmr.lib.models import load_tokenhmr
from .tokenhmr.lib.utils import recursive_to
from .tokenhmr.lib.utils.render_openpose import render_openpose
from .tokenhmr.lib.datasets.vitdet_dataset import (
    ViTDetDataset,
    DEFAULT_MEAN,
    DEFAULT_STD,
)
from .tokenhmr.lib.utils.renderer import Renderer, cam_crop_to_full

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

os.environ["PYOPENGL_PLATFORM"] = "egl"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


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
            "checkpoint": "src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
            "model_config": "src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
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

        cfg_path = "src/human_mesh/TokenHMR/tokenhmr/lib/configs/cascade_mask_rcnn_vitdet_h_75ep.py"
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

    def generate_mesh_from_frames(self, input_folder_path, out_dir):

        metadata_path = os.path.join(input_folder_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        fps = metadata["fps"] if "fps" in metadata else 24

        os.makedirs(out_dir, exist_ok=True)

        # Make list to collect masked real frames
        masked_real_mesh_frames = []
        masked_real_bbox_frames = []
        masked_real_hybrid_frames =[]
        mesh_frames = []  # Add list to collect mesh renderings
        overlay_frames = []  # Add list to collect overlay frames

        # Iterate over all images in folder
        for img_path in tqdm.tqdm(sorted(list(Path(input_folder_path).glob("*.jpg")) + list(Path(input_folder_path).glob("*.png")))):
            print(f"Processing {img_path}..")
            img_cv2 = cv2.imread(str(img_path))

            # Detect humans in image
            det_out = self.detector(img_cv2)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

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

                dict_info = capture_dict_info(out, name="model_out")
                save_json_path = os.path.join(out_dir, f'{img_path.stem}_info.json')
                with open(save_json_path, 'w') as f:
                    json.dump(dict_info, f, indent=4)

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
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
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

                    mask_save_path = os.path.join(out_dir, f'{img_fn}_{person_id}_mask.png')
                    cv2.imwrite(mask_save_path, body_mask * 255)

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

                    out_img_path = os.path.join(out_dir, f"{img_fn}_{person_id}.png")
                    cv2.imwrite(
                        out_img_path,
                        255 * final_img[:, :, ::-1],
                    )

                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                    if self.config["save_mesh"]:
                        camera_translation = cam_t.copy()
                        tmesh = self.renderer.vertices_to_trimesh(
                            verts, camera_translation, LIGHT_BLUE
                        )
                        mesh_out_path = os.path.join(
                            out_dir, f"{img_fn}_{person_id}.obj"
                        )
                        tmesh.export(mesh_out_path)

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
                cv2.imwrite(os.path.join(out_dir, f'{img_fn}_all.png'), (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8))
                
                # Save mesh frame for video
                mesh_frame = (255 * cam_view[:, :, :3]).astype(np.uint8)
                mesh_frames.append(mesh_frame)
                
                # Save overlay frame for video - keep in RGB format
                overlay_frame = (255 * input_img_overlay).astype(np.uint8)
                overlay_frames.append(overlay_frame)

        # After full loop
        if len(masked_real_bbox_frames) >= 2:
            print("\n✅ Computing motion from masked real frames...")

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

            # === Compute motions separately ===
            motion_mesh_rgb = compute_motion(masked_real_mesh_frames)
            motion_bbox_rgb = compute_motion(masked_real_bbox_frames)
            motion_hybrid_rgb = compute_motion(masked_real_hybrid_frames)

            # === Save motion videos ===
            export_to_video(motion_mesh_rgb, os.path.join(out_dir, "motion_masked_mesh.mp4"), fps=fps)
            export_to_video(motion_bbox_rgb, os.path.join(out_dir, "motion_masked_bbox.mp4"), fps=fps)
            export_to_video(motion_hybrid_rgb, os.path.join(out_dir, "motion_masked_hybrid.mp4"), fps=fps)

            # === Save mesh videos ===
            if len(mesh_frames) > 0:
                print("\n Creating mesh videos...")
                # Save mesh-only video
                mesh_frames_rgb = [(f.astype(np.float32) / 255.0) for f in mesh_frames]
                export_to_video(mesh_frames_rgb, os.path.join(out_dir, "mesh_renderings.mp4"), fps=fps)
                
                # Save overlay video (mesh on original frames)
                overlay_frames_rgb = [(f.astype(np.float32) / 255.0) for f in overlay_frames]
                export_to_video(overlay_frames_rgb, os.path.join(out_dir, "mesh_overlay.mp4"), fps=fps)

        print("\n Done!")

        # === Side-by-side-by-side video ===
        print("\n Creating side-by-side-by-side videos...")

        real_frames_paths = sorted(list(Path(input_folder_path).glob('*.png')) + list(Path(input_folder_path).glob('*.jpg')))
        real_frames = []
        for path in real_frames_paths:
            frame = imageio.imread(path)
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            frame = frame.astype(np.float32) / 255.0
            real_frames.append(frame)

        masked_mesh_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in masked_real_mesh_frames]
        masked_bbox_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in masked_real_bbox_frames]
        masked_hybrid_rgb = [(np.repeat(f[..., None], 3, axis=2) / 255.0).astype(np.float32) for f in masked_real_hybrid_frames]

        # === Align all lengths ===
        min_len = min(len(real_frames), len(masked_mesh_rgb), len(masked_bbox_rgb), len(masked_hybrid_rgb))
        real_frames = real_frames[:min_len]
        masked_mesh_rgb = masked_mesh_rgb[:min_len]
        masked_bbox_rgb = masked_bbox_rgb[:min_len]
        masked_hybrid_rgb = masked_hybrid_rgb[:min_len]

        # === Save separate combined videos ===
        def export_combined(real_frames, masked_frames, motion_frames, save_path):
            combined_frames = []
            for real_f, masked_f, motion_f in zip(real_frames, masked_frames, motion_frames):
                combined = np.concatenate([real_f, masked_f, motion_f], axis=1)
                combined_frames.append(combined)
            export_to_video(combined_frames, save_path, fps=fps)

        export_combined(real_frames, masked_mesh_rgb, motion_mesh_rgb, os.path.join(
            out_dir, "combined_masked_mesh.mp4"
        ))
        export_combined(real_frames, masked_bbox_rgb, motion_bbox_rgb, os.path.join(
            out_dir, "combined_masked_bbox.mp4"
        ))
        

if __name__ == "__main__":
    print("TokenHMR Mesh Generator")
