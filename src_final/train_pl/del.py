import logging
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate

from detectron2.data.datasets import register_coco_instances
import torch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T
import cv2
import numpy as np
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
import time
from detectron2.data.detection_utils import read_image

img = cv2.imread("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c04/frame_000002.jpg")
img = img[:, :, ::-1]  # BGR to RGB

# img = T.ResizeShortestEdge(short_edge_length=800, max_size=1333).get_transform(img).apply_image(img)

# img_tensor = torch.as_tensor(img.astype("float32").transpose(2,0,1))



# config_file = '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/human_mesh/TokenHMR/tokenhmr/lib/configs/cascade_mask_rcnn_vitdet_h_75ep.py'

# cfg = LazyConfig.load(config_file)


# cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

# for i in range(3):
#     cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25


# model = instantiate(cfg.model)
# model.to('cuda')
# model.eval()
# DetectionCheckpointer(model).load(cfg.train.init_checkpoint)



# with torch.no_grad():
#     outputs = model([{'image': img_tensor}])[0]
    
# visoutput = None
# visualizer = Visualizer(img, instance_mode = ColorMode.IMAGE)

# instances = outputs['instances'].to('cpu')
# print(instances)
# visoutput = visualizer.draw_instance_predictions(predictions = instances)

# ## Save file
# visoutput.save('output.jpg')

import torch
import numpy as np
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog

cfg = LazyConfig.load( '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/human_mesh/TokenHMR/tokenhmr/lib/configs/cascade_mask_rcnn_vitdet_h_75ep.py')
cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl" # replace with the path were you have your model
metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids
classes = metadata.thing_classes

model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

model.eval()

filename = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src_final/train_pl/vis_0002.png"
# image = Image.open(filename)
# original_image = np.array(image)
# image = np.array(image, dtype=np.uint8)
# image = np.moveaxis(image, -1, 0) # the model expects the image to be in channel first format

# with torch.inference_mode():
#     output = model([{'image': torch.from_numpy(image)}])[0]

cpu_device = torch.device("cpu")
metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names)  # to get labels from ids
aug = T.ResizeShortestEdge(short_edge_length=cfg.model.backbone.net.img_size, max_size=cfg.model.backbone.net.img_size)

original_image = read_image(filename, format="BGR")
height, width = original_image.shape[:2]
image = aug.get_transform(original_image).apply_image(original_image)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

inputs = {"image": image, "height": height, "width": width}

with torch.inference_mode():
    predictions = model([inputs])[0]

img = cv2.imread(filename)
visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)

instances = predictions["instances"].to(cpu_device)

visualized_output = visualizer.draw_instance_predictions(instances)
out_filename = f"del.png"
visualized_output.save(out_filename)