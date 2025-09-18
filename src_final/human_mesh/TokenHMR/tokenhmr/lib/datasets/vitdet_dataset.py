from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

# class ViTDetDataset(torch.utils.data.Dataset):

#     def __init__(self,
#                  cfg: CfgNode,
#                  img_cv2: np.array,
#                  boxes: np.array,
#                  train: bool = False,
#                  **kwargs):
#         super().__init__()
#         self.cfg = cfg
#         self.img_cv2 = img_cv2
#         # self.boxes = boxes

#         assert train == False, "ViTDetDataset is only for inference"
#         self.train = train
#         self.img_size = cfg.MODEL.IMAGE_SIZE
#         self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
#         self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

#         # Preprocess annotations
#         boxes = boxes.astype(np.float32)
#         self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
#         self.scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
#         self.personid = np.arange(len(boxes), dtype=np.int32)

#     def __len__(self) -> int:
#         return len(self.personid)

#     def __getitem__(self, idx: int) -> Dict[str, np.array]:

#         center = self.center[idx].copy()
#         center_x = center[0]
#         center_y = center[1]

#         scale = self.scale[idx]
#         BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
#         bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

#         patch_width = patch_height = self.img_size

#         # 3. generate image patch
#         # if use_skimage_antialias:
#         cvimg = self.img_cv2.copy()
#         if True:
#             # Blur image to avoid aliasing artifacts
#             downsampling_factor = ((bbox_size*1.0) / patch_width)
#             # print(f'{downsampling_factor=}')
#             downsampling_factor = downsampling_factor / 2.0
#             if downsampling_factor > 1.1:
#                 cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


#         img_patch_cv, trans = generate_image_patch_cv2(cvimg,
#                                                     center_x, center_y,
#                                                     bbox_size, bbox_size,
#                                                     patch_width, patch_height,
#                                                     False, 1.0, 0,
#                                                     border_mode=cv2.BORDER_CONSTANT)
#         img_patch_cv = img_patch_cv[:, :, ::-1]
#         img_patch = convert_cvimg_to_tensor(img_patch_cv)

#         # apply normalization
#         for n_c in range(min(self.img_cv2.shape[2], 3)):
#             img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

#         item = {
#             'img': img_patch,
#             'personid': int(self.personid[idx]),
#         }
#         item['box_center'] = self.center[idx].copy()
#         item['box_size'] = bbox_size
#         item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
#         return item

class ViTDetDataset(torch.utils.data.Dataset):
    """
    Supports:
      - Single frame: img_cv2: np.ndarray(H,W,3); boxes: (N,4)
      - Multi frame : img_cv2: List[np.ndarray(H,W,3)] or np.ndarray(T,H,W,3);
                      boxes:   List[np.ndarray(M_i,4)] or np.ndarray(T,M,4)
    Each __getitem__ returns one cropped/normalized patch for one person in one frame.
    """
    def __init__(self,
                 cfg: CfgNode,
                 img_cv2,
                 boxes,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        assert train is False, "ViTDetDataset is only for inference"
        self.cfg = cfg
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        self.std  = 255. * np.array(self.cfg.MODEL.IMAGE_STD,  dtype=np.float32)

        # ---- Normalize input shapes to multi-frame lists ----
        # Frames -> list of np.ndarray(H,W,3)
        if isinstance(img_cv2, np.ndarray) and img_cv2.ndim == 3:
            self.frames = [img_cv2]                         # [T=1]
        elif isinstance(img_cv2, np.ndarray) and img_cv2.ndim == 4:
            self.frames = [img_cv2[t] for t in range(img_cv2.shape[0])]
        elif isinstance(img_cv2, (list, tuple)):
            self.frames = list(img_cv2)
        else:
            raise TypeError("img_cv2 must be HxWx3, TxHxWx3, or list of HxWx3 arrays")

        # Boxes -> list of (Mi,4) float32 arrays (one per frame)
        def to_boxes_list(bx):
            if isinstance(bx, np.ndarray):
                if bx.ndim == 2 and bx.shape[-1] == 4:
                    return [bx]                             # single frame
                if bx.ndim == 3 and bx.shape[-1] == 4:
                    return [bx[t] for t in range(bx.shape[0])]
                raise ValueError("boxes ndarray must be (N,4) or (T,N,4)")
            if isinstance(bx, (list, tuple)):
                out = []
                for b in bx:
                    b = np.asarray(b, dtype=np.float32)
                    if b.ndim != 2 or b.shape[-1] != 4:
                        raise ValueError("each boxes entry must be (Mi,4)")
                    out.append(b)
                return out
            raise TypeError("boxes must be ndarray or list/tuple of (Mi,4) arrays")

        boxes_list = to_boxes_list(boxes)
        if len(boxes_list) != len(self.frames):
            raise ValueError(f"#frames ({len(self.frames)}) != #boxes lists ({len(boxes_list)})")

        # Precompute per-frame center/scale and build flat index (frame_idx, person_idx)
        self.frame_meta = []  # list of dict per frame
        self.flat_index = []  # list of (t, i) pairs
        for t, (frame, b) in enumerate(zip(self.frames, boxes_list)):
            b = b.astype(np.float32)
            center = (b[:, 2:4] + b[:, 0:2]) / 2.0
            scale  = (b[:, 2:4] - b[:, 0:2]) / 200.0  # as in original
            self.frame_meta.append({
                "center": center,  # (Mi,2)
                "scale":  scale,   # (Mi,2) (width/200, height/200)
            })
            for i in range(len(b)):
                self.flat_index.append((t, i))

        # Backward-compat fields (for code that inspects length/personid)
        self.personid = np.arange(len(self.flat_index), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Resolve which (frame, person) this index corresponds to
        t, i = self.flat_index[idx]
        frame = self.frames[t]
        center = self.frame_meta[t]["center"][i].copy()
        scale  = self.frame_meta[t]["scale"][i].copy()

        center_x, center_y = float(center[0]), float(center[1])
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_wh = expand_to_aspect_ratio(scale * 200.0, target_aspect_ratio=BBOX_SHAPE)
        bbox_size = float(bbox_wh.max())

        patch_width = patch_height = self.img_size

        # Anti-aliasing blur (same logic as before)
        cvimg = frame.copy()
        downsampling_factor = (bbox_size / patch_width) / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2,
                             channel_axis=2, preserve_range=True)

        # Crop/warp to patch
        img_patch_cv, trans = generate_image_patch_cv2(
            cvimg, center_x, center_y,
            bbox_size, bbox_size,
            patch_width, patch_height,
            False, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT
        )
        # BGR->RGB as in original
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)  # (C,H,W) float32 in [0..255]

        # Normalize
        C = min(frame.shape[2], 3)
        for c in range(C):
            img_patch[c, :, :] = (img_patch[c, :, :] - self.mean[c]) / self.std[c]

        item = {
            'img': img_patch,                 # torch.FloatTensor (C,H,W)
            'personid': int(i),               # index within frame t
            'frame_index': int(t),            # which frame
            'box_center': center,             # (2,)
            'box_size': bbox_size,            # scalar
            'img_size': np.array([cvimg.shape[1], cvimg.shape[0]], dtype=np.float32)
        }
        return item