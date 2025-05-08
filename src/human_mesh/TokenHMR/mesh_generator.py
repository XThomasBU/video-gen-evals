from .tokenhmr.lib.models import load_tokenhmr
from .tokenhmr.lib.utils import recursive_to
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

os.environ["PYOPENGL_PLATFORM"] = "egl"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


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

    def generate_mesh_from_frames(self, folder_path):
        out_dir = os.path.join(folder_path, "tokenhmr_mesh")

        os.makedirs(out_dir, exist_ok=True)

        # Iterate over all images in folder
        for img_path in tqdm.tqdm(sorted(Path(folder_path).glob("*.png"))):
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

                input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
                input_img = np.concatenate(
                    [input_img, np.ones_like(input_img[:, :, :1])], axis=2
                )  # alpha channel
                input_img_overlay = (
                    input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                    + cam_view[:, :, :3] * cam_view[:, :, 3:]
                )

                out_overlay_path = os.path.join(out_dir, f"{img_fn}_all.png")
                cv2.imwrite(
                    out_overlay_path,
                    255 * input_img_overlay[:, :, ::-1],
                )


if __name__ == "__main__":
    print("TokenHMR Mesh Generator")
