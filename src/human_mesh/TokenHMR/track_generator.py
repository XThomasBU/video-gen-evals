import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
# from hmr2.datasets.utils import expand_bbox_to_aspect_ratio
from pycocotools import mask as mask_utils
from scenedetect import AdaptiveDetector, detect
from sklearn.linear_model import Ridge

from phalp.configs.base import CACHE_DIR
from phalp.external.deep_sort_ import nn_matching
from phalp.external.deep_sort_.detection import Detection
from phalp.external.deep_sort_.tracker import Tracker
from phalp.models.hmar import HMAR
from phalp.models.predictor import Pose_transformer_v2
from phalp.utils import get_pylogger
from phalp.utils.io import IO_Manager
from phalp.utils.utils import (convert_pkl, get_prediction_interval,
                               progress_bar, smpl_to_pose_camera_vector)
from phalp.utils.utils_dataset import process_image, process_mask
from phalp.utils.utils_detectron2 import (DefaultPredictor_Lazy,
                                          DefaultPredictor_with_RPN)
from phalp.utils.utils_download import cache_url
from phalp.visualize.postprocessor import Postprocessor
from phalp.visualize.visualizer import Visualizer

log = get_pylogger(__name__)

class PHALP(nn.Module):

    def __init__(self, cfg):
        super(PHALP, self).__init__()

        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        self.io_manager = IO_Manager(self.cfg)

        # download wights and configs from Google Drive
        self.cached_download_from_drive()
        
        # setup HMR, and pose_predictor. Override this function to use your own model
        self.setup_hmr()

        # setup temporal pose predictor
        self.setup_predictor()
        
        # setup Detectron2, override this function to use your own model
        self.setup_detectron2()
        
        # create a visualizer
        self.setup_visualizer()
        
        # move to device
        self.to(self.device)
        
        # train or eval
        self.train() if(self.cfg.train) else self.eval()
        
        # create nessary directories
        self.default_setup()
        
    def setup_hmr(self):
        log.info("Loading HMAR model...")
        self.HMAR = HMAR(self.cfg)
        self.HMAR.load_weights(self.cfg.hmr.hmar_path)

    def setup_predictor(self):
        log.info("Loading Predictor model...")
        self.pose_predictor = Pose_transformer_v2(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)
        
    def setup_detectron2(self):
        log.info("Loading Detection model...")
        if self.cfg.phalp.detector == 'maskrcnn':
            self.detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            self.detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            self.detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            self.detector       = DefaultPredictor_Lazy(self.detectron2_cfg)
            self.class_names    = self.detector.metadata.get('thing_classes')
        elif self.cfg.phalp.detector == 'vitdet':
            from detectron2.config import LazyConfig
            import phalp
            cfg_path = Path(phalp.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            self.detectron2_cfg = LazyConfig.load(str(cfg_path))
            self.detectron2_cfg.train.init_checkpoint = 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
            for i in range(3):
                self.detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
            self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)
        else:
            raise ValueError(f"Detector {self.cfg.phalp.detector} not supported")        

        # for predicting masks with only bounding boxes, e.g. for running on ground truth tracks
        self.setup_detectron2_with_RPN()
        # TODO: make this work with DefaultPredictor_Lazy
        
    def setup_detectron2_with_RPN(self):
        self.detectron2_cfg = get_cfg()
        self.detectron2_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))   
        self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.4
        self.detectron2_cfg.MODEL.WEIGHTS   = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.detectron2_cfg.MODEL.META_ARCHITECTURE =  "GeneralizedRCNN_with_proposals"
        self.detector_x = DefaultPredictor_with_RPN(self.detectron2_cfg)
        
    def setup_deepsort(self):
        log.info("Setting up DeepSort...")
        metric  = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])  
        
    def setup_visualizer(self):
        log.info("Setting up Visualizer...")
        self.visualizer = Visualizer(self.cfg, self.HMAR)
    
    def setup_postprocessor(self):
        # by default this will not be initialized
        self.postprocessor = Postprocessor(self.cfg, self)

    def default_setup(self):
        # create subfolders for saving additional results
        try:
            os.makedirs(self.cfg.video.output_dir + '/results', exist_ok=True)  
            os.makedirs(self.cfg.video.output_dir + '/results_tracks', exist_ok=True)  
            os.makedirs(self.cfg.video.output_dir + '/_TMP', exist_ok=True)  
            os.makedirs(self.cfg.video.output_dir + '/_DEMO', exist_ok=True)  
        except: 
            pass
        
    def track(self):
        
        eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
        history_keys    = ['appe', 'loca', 'pose', 'uv'] if self.cfg.render.enable else []
        prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if self.cfg.render.enable else []
        extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'class_name', 'conf', 'annotations']
        extra_keys_2    = ['smpl', 'camera', 'camera_bbox', '3d_joints', '2d_joints', 'mask', 'extra_data']
        history_keys    = history_keys + extra_keys_1 + extra_keys_2
        visual_store_   = eval_keys + history_keys + prediction_keys
        tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
        
        # process the source video and return a list of frames
        # source can be a video file, a youtube link or a image folder
        io_data = self.io_manager.get_frames_from_source()
        list_of_frames, additional_data = io_data['list_of_frames'], io_data['additional_data']
        self.cfg.video_seq = io_data['video_name']
        pkl_path = self.cfg.video.output_dir + '/results/' + self.cfg.track_dataset + "_" + str(self.cfg.video_seq) + '.pkl'
        video_path = self.cfg.video.output_dir + '/' + self.cfg.base_tracker + '_' + str(self.cfg.video_seq) + '.mp4'
        
        # check if the video is already processed                                  
        if(not(self.cfg.overwrite) and os.path.isfile(pkl_path)): 
            return 0
        
        # eval mode
        self.eval()
        
        # setup rendering, deep sort and directory structure
        self.setup_deepsort()
        self.default_setup()
        
        log.info("Saving tracks at : " + self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq))
        
        try: 
            
            list_of_frames = list_of_frames if self.cfg.phalp.start_frame==-1 else list_of_frames[self.cfg.phalp.start_frame:self.cfg.phalp.end_frame]
            list_of_shots = self.get_list_of_shots(list_of_frames)
            
            tracked_frames = []
            final_visuals_dic = {}
            
            for t_, frame_name in progress_bar(enumerate(list_of_frames), description="Tracking : " + self.cfg.video_seq, total=len(list_of_frames), disable=False):
                
                image_frame               = self.io_manager.read_frame(frame_name)
                img_height, img_width, _  = image_frame.shape
                new_image_size            = max(img_height, img_width)
                top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
                measurments               = [img_height, img_width, new_image_size, left, top]
                self.cfg.phalp.shot       = 1 if t_ in list_of_shots else 0

                if(self.cfg.render.enable):
                    # reset the renderer
                    # TODO: add a flag for full resolution rendering
                    self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
                    self.visualizer.reset_render(self.cfg.render.res*self.cfg.render.up_scale)
                
                ############ detection ##############
                pred_bbox, pred_bbox_pad, pred_masks, pred_scores, pred_classes, gt_tids, gt_annots = self.get_detections(image_frame, frame_name, t_, additional_data, measurments)

                ############ Run EXTRA models to attach to the detections ##############
                extra_data = self.run_additional_models(image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots)
                
                ############ HMAR ##############
                detections = self.get_human_features(image_frame, pred_masks, pred_bbox, pred_bbox_pad, pred_scores, frame_name, pred_classes, t_, measurments, gt_tids, gt_annots, extra_data)

                ############ tracking ##############
                self.tracker.predict()
                self.tracker.update(detections, t_, frame_name, self.cfg.phalp.shot)

                ############ record the results ##############
                final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': self.cfg.phalp.shot, 'frame_path': frame_name})
                if(self.cfg.render.enable): final_visuals_dic[frame_name]['frame'] = image_frame
                for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
                
                ############ record the track states (history and predictions) ##############
                for tracks_ in self.tracker.tracks:
                    if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                    if(not(tracks_.is_confirmed())): continue
                    
                    track_id        = tracks_.track_id
                    track_data_hist = tracks_.track_data['history'][-1]
                    track_data_pred = tracks_.track_data['prediction']

                    final_visuals_dic[frame_name]['tid'].append(track_id)
                    final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                    final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                    for hkey_ in history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                    for pkey_ in prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])

                    if(tracks_.time_since_update==0):
                        final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                        final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                        
                        if(tracks_.hits==self.cfg.phalp.n_init):
                            for pt in range(self.cfg.phalp.n_init-1):
                                track_data_hist_ = tracks_.track_data['history'][-2-pt]
                                track_data_pred_ = tracks_.track_data['prediction']
                                frame_name_      = tracked_frames[-2-pt]
                                final_visuals_dic[frame_name_]['tid'].append(track_id)
                                final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                                final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                                final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                                final_visuals_dic[frame_name_]['tracked_time'].append(0)

                                for hkey_ in history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                                for pkey_ in prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])

                ############ save the video ##############
                if(self.cfg.render.enable and t_>=self.cfg.phalp.n_init):                    
                    d_ = self.cfg.phalp.n_init+1 if(t_+1==len(list_of_frames)) else 1
                    for t__ in range(t_, t_+d_):

                        frame_key = list_of_frames[t__-self.cfg.phalp.n_init]
                        rendered_, f_size = self.visualizer.render_video(final_visuals_dic[frame_key])      

                        # save the rendered frame
                        self.io_manager.save_video(video_path, rendered_, f_size, t=t__-self.cfg.phalp.n_init)

                        # delete the frame after rendering it
                        del final_visuals_dic[frame_key]['frame']
                        
                        # delete unnecessary keys
                        for tkey_ in tmp_keys_:  
                            del final_visuals_dic[frame_key][tkey_] 

            joblib.dump(final_visuals_dic, pkl_path, compress=3)
            self.io_manager.close_video()
            if(self.cfg.use_gt): joblib.dump(self.tracker.tracked_cost, self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq) + '_' + str(self.cfg.phalp.start_frame) + '_distance.pkl')
            
            return final_visuals_dic, pkl_path
            
        except Exception as e: 
            print(e)
            print(traceback.format_exc())         

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        
        if(frame_name in additional_data.keys()):
            
            img_height, img_width, new_image_size, left, top = measurments
            
            gt_bbox = additional_data[frame_name]["gt_bbox"]
            if(len(additional_data[frame_name]["extra_data"]['gt_track_id']) > 0):
                ground_truth_track_id = additional_data[frame_name]["extra_data"]['gt_track_id']
            else:
                ground_truth_track_id = [-1 for i in range(len(gt_bbox))]

            if(len(additional_data[frame_name]["extra_data"]['gt_class']) > 0):
                ground_truth_annotations = additional_data[frame_name]["extra_data"]['gt_class']
            else:
                ground_truth_annotations = [[] for i in range(len(gt_bbox))]
                
            inst = Instances((img_height, img_width))
            bbox_array   = []
            class_array  = []
            scores_array = []

            # for ava bbox format  
            # for bbox_ in gt_bbox:
            #     x1 = bbox_[0] * img_width
            #     y1 = bbox_[1] * img_height
            #     x2 = bbox_[2] * img_width
            #     y2 = bbox_[3] * img_height

            # for posetrack bbox format
            for bbox_ in gt_bbox:
                x1 = bbox_[0]
                y1 = bbox_[1]
                x2 = bbox_[2] + x1
                y2 = bbox_[3] + y1

                bbox_array.append([x1, y1, x2, y2])
                class_array.append(0)
                scores_array.append(1)
                    
            bbox_array          = np.array(bbox_array)
            class_array         = np.array(class_array)
            box                 = Boxes(torch.as_tensor(bbox_array))
            inst.pred_boxes     = box
            inst.pred_classes   = torch.as_tensor(class_array)
            inst.scores         = torch.as_tensor(scores_array)
            
            outputs_x           = self.detector_x.predict_with_bbox(image, inst)                 
            instances_x         = outputs_x['instances']
            instances_people    = instances_x[instances_x.pred_classes==0]
            
            pred_bbox   = instances_people.pred_boxes.tensor.cpu().numpy()
            pred_masks  = instances_people.pred_masks.cpu().numpy()
            pred_scores = instances_people.scores.cpu().numpy()
            pred_classes= instances_people.pred_classes.cpu().numpy()
                                    
        else:
            outputs     = self.detector(image)   
            instances   = outputs['instances']
            instances   = instances[instances.pred_classes==0]
            instances   = instances[instances.scores>self.cfg.phalp.low_th_c]

            pred_bbox   = instances.pred_boxes.tensor.cpu().numpy()
            pred_masks  = instances.pred_masks.cpu().numpy()
            pred_scores = instances.scores.cpu().numpy()
            pred_classes= instances.pred_classes.cpu().numpy()
            
            ground_truth_track_id = [1 for i in list(range(len(pred_scores)))]
            ground_truth_annotations = [[] for i in list(range(len(pred_scores)))]

        return pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, ground_truth_track_id, ground_truth_annotations

    def get_croped_image(self, image, bbox, bbox_pad, seg_mask):
        
        # Encode the mask for storing, borrowed from tao dataset
        # https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/detectron2_infer.py
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order='F', dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for rle in rles: 
            rle["counts"] = rle["counts"].decode("utf-8")
            
        seg_mask = seg_mask.astype(int)*255
        if(len(seg_mask.shape)==2):
            seg_mask = np.expand_dims(seg_mask, 2)
            seg_mask = np.repeat(seg_mask, 3, 2)
        
        center_      = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
        scale_       = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])

        center_pad   = np.array([(bbox_pad[2] + bbox_pad[0])/2, (bbox_pad[3] + bbox_pad[1])/2])
        scale_pad    = np.array([(bbox_pad[2] - bbox_pad[0]), (bbox_pad[3] - bbox_pad[1])])
        mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_pad, 1.0*np.max(scale_pad))
        image_tmp    = process_image(image, center_pad, 1.0*np.max(scale_pad))

        # bbox_        = expand_bbox_to_aspect_ratio(bbox, target_aspect_ratio=(192,256))
        # center_x     = np.array([(bbox_[2] + bbox_[0])/2, (bbox_[3] + bbox_[1])/2])
        # scale_x      = np.array([(bbox_[2] - bbox_[0]), (bbox_[3] - bbox_[1])])
        # mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_x, 1.0*np.max(scale_x))
        # image_tmp    = process_image(image, center_x, 1.0*np.max(scale_x))
        
        masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
        
        return masked_image, center_, scale_, rles, center_pad, scale_pad
    
    def run_additional_models(self, image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots):
        return list(range(len(pred_scores)))

    def get_human_features(self, image, seg_mask, bbox, bbox_pad, score, frame_name, cls_id, t_, measurments, gt=1, ann=None, extra_data=None):
        NPEOPLE = len(score)

        if(NPEOPLE==0): return []

        img_height, img_width, new_image_size, left, top = measurments                
        ratio = 1.0/int(new_image_size)*self.cfg.render.res
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []
        for p_ in range(NPEOPLE):
            if bbox[p_][2]-bbox[p_][0]<self.cfg.phalp.small_w or bbox[p_][3]-bbox[p_][1]<self.cfg.phalp.small_h:
                continue
            masked_image, center_, scale_, rles, center_pad, scale_pad = self.get_croped_image(image, bbox[p_], bbox_pad[p_], seg_mask[p_])
            masked_image_list.append(masked_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            rles_list.append(rles)
            selected_ids.append(p_)
        
        if(len(masked_image_list)==0): return []

        masked_image_list = torch.stack(masked_image_list, dim=0)
        BS = masked_image_list.size(0)
        
        with torch.no_grad():
            extra_args      = {}
            hmar_out        = self.HMAR(masked_image_list.cuda(), **extra_args) 
            uv_vector       = hmar_out['uv_vector']
            appe_embedding  = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding  = appe_embedding.view(appe_embedding.shape[0], -1)
            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(hmar_out['pose_smpl'], hmar_out['pred_cam'],
                                                                                               center=(np.array(center_list) + np.array([left, top]))*ratio,
                                                                                               img_size=self.cfg.render.res,
                                                                                               scale=np.max(np.array(scale_list), axis=1, keepdims=True)*ratio)
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]
            
            if(self.cfg.phalp.pose_distance=="joints"):
                pose_embedding  = pred_joints.cpu().view(BS, -1)
            elif(self.cfg.phalp.pose_distance=="smpl"):
                pose_embedding = []
                for i in range(BS):
                    pose_embedding_  = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            pred_joints_2d_ = pred_joints_2d.reshape(BS,-1)/self.cfg.render.res
            pred_cam_ = pred_cam.view(BS, -1)
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()
            loca_embedding  = torch.cat((pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 1)
        
        # keeping it here for legacy reasons (T3DP), but it is not used.
        full_embedding    = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding.cpu()), 1)
        
        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                                "bbox"            : np.array([bbox[p_][0], bbox[p_][1], (bbox[p_][2] - bbox[p_][0]), (bbox[p_][3] - bbox[p_][1])]),
                                "mask"            : rles_list[i],
                                "conf"            : score[p_], 
                                
                                "appe"            : appe_embedding[i].cpu().numpy(), 
                                "pose"            : pose_embedding[i].numpy(), 
                                "loca"            : loca_embedding[i].cpu().numpy(), 
                                "uv"              : uv_vector[i].cpu().numpy(), 
                                
                                "embedding"       : full_embedding[i], 
                                "center"          : center_list[i],
                                "scale"           : scale_list[i],
                                "smpl"            : pred_smpl_params[i],
                                "camera"          : pred_cam_[i].cpu().numpy(),
                                "camera_bbox"     : hmar_out['pred_cam'][i].cpu().numpy(),
                                "3d_joints"       : pred_joints[i].cpu().numpy(),
                                "2d_joints"       : pred_joints_2d_[i].cpu().numpy(),
                                "size"            : [img_height, img_width],
                                "img_path"        : frame_name,
                                "img_name"        : frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                                "class_name"      : cls_id[p_],
                                "time"            : t_,

                                "ground_truth"    : gt[p_],
                                "annotations"     : ann[p_],
                                "extra_data"      : extra_data[p_] if extra_data is not None else None
                            }
            detection_data_list.append(Detection(detection_data))

        return detection_data_list
    
    def forward_for_tracking(self, vectors, attibute="A", time=1):
        
        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]
        
            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)
            
            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0) # (BS, 7, pose_dim)
                en_time          = en_time.unsqueeze(0) # (BS, 7)
                en_data          = en_data.unsqueeze(0) # (BS, 7, 6)
            
            with torch.no_grad():
                pose_pred, ava_pred, smpl_pred = self.pose_predictor.predict_next(en_pose, en_data, en_time, time)

                smpl_pred = {k: v.cpu().numpy() for k, v in smpl_pred.items()}
            
            return pose_pred.cpu(), ava_pred.cpu(), smpl_pred


        if(attibute=="L"):
            vectors_loca         = vectors[0]
            vectors_time         = vectors[1]
            vectors_conf         = vectors[2]

            en_loca              = torch.from_numpy(vectors_loca)
            en_time              = torch.from_numpy(vectors_time)
            en_conf              = torch.from_numpy(vectors_conf)
            time                 = torch.from_numpy(time)

            if(len(en_loca.shape)!=3):
                en_loca          = en_loca.unsqueeze(0)             
                en_time          = en_time.unsqueeze(0)             
            else:
                en_loca          = en_loca.permute(0, 1, 2)         

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy           = en_loca[:, :, :90]
            en_loca_xy           = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n            = en_loca[:, :, 90:]
            en_loca_n            = en_loca_n.view(BS, t_, 3, 3)

            new_en_loca_n = []
            for bs in range(BS):
                x0_                  = np.array(en_loca_xy[bs, :, 44, 0])
                y0_                  = np.array(en_loca_xy[bs, :, 44, 1])
                n_                   = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                t_                   = np.array(en_time[bs, :])

                loc_                 = torch.diff(en_time[bs, :], dim=0)!=0
                if(self.cfg.phalp.distance_type=="EQ_020" or self.cfg.phalp.distance_type=="EQ_021"):
                    loc_                 = 1
                else:
                    loc_                 = loc_.shape[0] - torch.sum(loc_)+1

                M = t_[:, np.newaxis]**[0, 1]
                time_ = 48 if time[bs]>48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi  = get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                x_p  = x_p[0]
                x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi  = get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                y_p  = y_p[0]
                y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi  = get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])
                
                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi/loc_, y_pi/loc_, np.exp(n_pi)/loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p
                
            new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
            xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt

    def get_uv_distance(self, t_uv, d_uv):
        t_uv         = torch.from_numpy(t_uv).cuda().float()
        d_uv         = torch.from_numpy(d_uv).cuda().float()
        d_mask       = d_uv[3:, :, :]>0.5
        t_mask       = t_uv[3:, :, :]>0.5
        
        mask_dt      = torch.logical_and(d_mask, t_mask)
        mask_dt      = mask_dt.repeat(4, 1, 1)
        mask_        = torch.logical_not(mask_dt)
        
        t_uv[mask_]  = 0.0
        d_uv[mask_]  = 0.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb        = t_emb.view(-1)/10**3
        d_emb        = d_emb.view(-1)/10**3
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy()/4/256/256/2

    def get_pose_distance(self, track_pose, detect_pose):
        """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`.""" 
        track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

        if(self.cfg.phalp.pose_distance=="smpl"):
            # remove additional dimension used for encoding location (last 3 elements)
            track_pose = track_pose[:, :-3]
            detect_pose = detect_pose[:, :-3]

        if len(track_pose) == 0 or len(detect_pose) == 0:
            return np.zeros((len(track_pose), len(detect_pose)))
        track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
        r2 = -2. * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))

        return r2

    def get_list_of_shots(self, list_of_frames):
        # https://github.com/Breakthrough/PySceneDetect
        list_of_shots    = []
        remove_tmp_video = False
        if(self.cfg.detect_shots):
            if(isinstance(list_of_frames[0], str)):
                # make a video if list_of_frames is frames
                video_tmp_name   = self.cfg.video.output_dir + "/_TMP/" + str(self.cfg.video_seq) + ".mp4"
                for ft_, fname_ in enumerate(list_of_frames):
                    im_ = cv2.imread(fname_)
                    if(ft_==0): 
                        video_file = cv2.VideoWriter(video_tmp_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, frameSize=(im_.shape[1], im_.shape[0]))
                    video_file.write(im_)
                video_file.release()
                remove_tmp_video = True
            elif(isinstance(list_of_frames[0], tuple)):
                video_tmp_name = list_of_frames[0][0]
            else:
                raise Exception("Unknown type of list_of_frames")
            
            # Detect scenes in a video using PySceneDetect.
            scene_list = detect(video_tmp_name, AdaptiveDetector())

            if(remove_tmp_video):
                os.system("rm " + video_tmp_name)

            for scene in scene_list:
                list_of_shots.append(scene[0].get_frames())
                list_of_shots.append(scene[1].get_frames())
            list_of_shots = np.unique(list_of_shots)
            list_of_shots = list_of_shots[1:-1]
            log.info("Detected shot change at frame"+ "s" * min(0,len(list_of_shots)-1) + ": " + ", ".join(map(str, list_of_shots)))

        return list_of_shots

    def cached_download_from_drive(self, additional_urls=None):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """
        
        os.makedirs(os.path.join(CACHE_DIR, "phalp"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/3D"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/weights"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/ava"), exist_ok=True)

        smpl_path = os.path.join(CACHE_DIR, "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl")

        if not os.path.exists(smpl_path):
            # We are downloading the SMPL model here for convenience. Please accept the license
            # agreement on the SMPL website: https://smpl.is.tue.mpg.
            os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/smpl"), exist_ok=True)
            os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

            convert_pkl('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('rm basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('mv basicModel_neutral_lbs_10_207_0_v1.0.0_p3.pkl ' + smpl_path)

        additional_urls = additional_urls if additional_urls is not None else {}
        download_files = {
            "head_faces.npy"           : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "mean_std.npy"             : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "smpl_mean_params.npz"     : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "SMPL_to_J19.pkl"          : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl", os.path.join(CACHE_DIR, "phalp/3D")],
            "texture.npz"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "bmap_256.npy"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/bmap_256.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "fmap_256.npy"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/fmap_256.npy", os.path.join(CACHE_DIR, "phalp/3D")],

            "hmar_v2_weights.pth"      : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.pth"       : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.yaml"      : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml", os.path.join(CACHE_DIR, "phalp/weights")],
            
            # data for ava dataset
            "ava_labels.pkl"           : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
            "ava_class_mapping.pkl"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
    
        } | additional_urls # type: ignore
        
        for file_name, url in download_files.items():
            if not os.path.exists(os.path.join(url[1], file_name)):
                print("Downloading file: " + file_name)
                # output = gdown.cached_download(url[0], os.path.join(url[1], file_name), fuzzy=True)
                output = cache_url(url[0], os.path.join(url[1], file_name))
                assert os.path.exists(os.path.join(url[1], file_name)), f"{output} does not exist"


import warnings
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.models.predictor.pose_transformer_v2 import *
# from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.external.deep_sort_.detection import Detection
from phalp.external.deep_sort_.tracker import *
from phalp.external.deep_sort_.track import *
from phalp.external.deep_sort_ import nn_matching
from diffusers.utils import export_to_video
import traceback
import os
import json
import joblib
from tqdm import tqdm
import imageio
import numpy as np
import torch
import cv2
import torch.nn as nn
import copy
from collections import deque

import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d
from phalp.utils.utils import (convert_pkl, get_prediction_interval,
                               progress_bar, smpl_to_pose_camera_vector)

warnings.filterwarnings('ignore')
log = get_pylogger(__name__)

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted   = 3

class Track:
    """
    Mark this track as missed (no association at the current time step).
    """

    def __init__(self, cfg, track_id, n_init, max_age, detection_data, detection_id=None, dims=None):
        self.cfg               = cfg
        self.track_id          = track_id
        self.hits              = 1
        self.age               = 1
        self.time_since_update = 0
        self.time_init         = detection_data["time"]
        self.state             = TrackState.Tentative            
        
        self._n_init           = n_init
        self._max_age          = max_age
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
        self.track_data        = {"history": deque(maxlen=self.cfg.phalp.track_history) , "prediction":{}}
        for _ in range(self.cfg.phalp.track_history):
            self.track_data["history"].append(detection_data)
            
        self.track_data['prediction']['appe'] = deque([detection_data['appe']], maxlen=self.cfg.phalp.n_init+1)
        self.track_data['prediction']['loca'] = deque([detection_data['loca']], maxlen=self.cfg.phalp.n_init+1)
        self.track_data['prediction']['pose'] = deque([detection_data['pose']], maxlen=self.cfg.phalp.n_init+1)
        self.track_data['prediction']['uv']   = deque([copy.deepcopy(detection_data['uv'])], maxlen=self.cfg.phalp.n_init+1)

        # if the track is initialized by detection with annotation, then we set the track state to confirmed
        if len(detection_data['annotations'])>0:
            self.state = TrackState.Confirmed      

    def predict(self, phalp_tracker, increase_age=True):
        if(increase_age):
            self.age += 1; self.time_since_update += 1
            
    def add_predicted(self, appe=None, pose=None, loca=None, uv=None):
        appe_predicted = copy.deepcopy(appe.numpy()) if(appe is not None) else copy.deepcopy(self.track_data['history'][-1]['appe'])
        loca_predicted = copy.deepcopy(loca.numpy()) if(loca is not None) else copy.deepcopy(self.track_data['history'][-1]['loca'])
        pose_predicted = copy.deepcopy(pose.numpy()) if(pose is not None) else copy.deepcopy(self.track_data['history'][-1]['pose'])
        
        self.track_data['prediction']['appe'].append(appe_predicted)
        self.track_data['prediction']['loca'].append(loca_predicted)
        self.track_data['prediction']['pose'].append(pose_predicted)

    def update(self, detection, detection_id, shot):             

        self.track_data["history"].append(copy.deepcopy(detection.detection_data))
        if(shot==1): 
            for tx in range(self.cfg.phalp.track_history):
                self.track_data["history"][-1-tx]['loca'] = copy.deepcopy(detection.detection_data['loca'])

        if("T" in self.cfg.phalp.predict):
            mixing_alpha_                      = self.cfg.phalp.alpha*(detection.detection_data['conf']**2)
            ones_old                           = self.track_data['prediction']['uv'][-1][3:, :, :]==1
            ones_new                           = self.track_data['history'][-1]['uv'][3:, :, :]==1
            ones_old                           = np.repeat(ones_old, 3, 0)
            ones_new                           = np.repeat(ones_new, 3, 0)
            ones_intersect                     = np.logical_and(ones_old, ones_new)
            ones_union                         = np.logical_or(ones_old, ones_new)
            good_old_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_old)
            good_new_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_new)
            new_rgb_map                        = np.zeros((3, 256, 256))
            new_mask_map                       = np.zeros((1, 256, 256))-1
            new_mask_map[ones_union[:1, :, :]] = 1.0
            new_rgb_map[ones_intersect]        = (1-mixing_alpha_)*self.track_data['prediction']['uv'][-1][:3, :, :][ones_intersect] + mixing_alpha_*self.track_data['history'][-1]['uv'][:3, :, :][ones_intersect]
            new_rgb_map[good_old_ones]         = self.track_data['prediction']['uv'][-1][:3, :, :][good_old_ones] 
            new_rgb_map[good_new_ones]         = self.track_data['history'][-1]['uv'][:3, :, :][good_new_ones] 
            self.track_data['prediction']['uv'].append(np.concatenate((new_rgb_map , new_mask_map), 0))
        else:
            self.track_data['prediction']['uv'].append(self.track_data['history'][-1]['uv'])
            
        
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # if the detection has annotation, then we set the track state to confirmed
        if len(detection.detection_data['annotations'])>0:
            self.state = TrackState.Confirmed
        
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def smooth_bbox(self, bbox):
        kernel_size = 5
        sigma       = 3
        bbox        = np.array(bbox)
        smoothed    = np.array([signal.medfilt(param, kernel_size) for param in bbox.T]).T
        out         = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T
        return list(out)

class My_Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, cfg, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.cfg              = cfg
        self.metric           = metric
        self.max_age          = max_age
        self.n_init           = n_init
        self.tracks           = []
        self._next_id         = 1
        self.tracked_cost     = {}
        self.phalp_tracker    = phalp_tracker
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.phalp_tracker, increase_age=True)

    def update(self, detections, frame_t, image_name, shot):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)
        self.tracked_cost[frame_t] = [statistics[0], matches, unmatched_tracks, unmatched_detections, statistics[1], statistics[2], statistics[3], statistics[4]] 
        if(self.cfg.verbose): print(np.round(np.array(statistics[0]), 2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], detection_idx, shot)
        self.accumulate_vectors([i[0] for i in matches], features=self.cfg.phalp.predict)
 
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.accumulate_vectors(unmatched_tracks, features=self.cfg.phalp.predict)
    
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)
            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_tentative()]
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                         
            appe_features += [track.track_data['prediction']['appe'][-1]]
            loca_features += [track.track_data['prediction']['loca'][-1]]
            pose_features += [track.track_data['prediction']['pose'][-1]]
            uv_maps       += [track.track_data['prediction']['uv'][-1]]
            targets       += [track.track_id]
            
            
        self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        
        return matches
        
    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb          = np.array([dets[i].detection_data['appe'] for i in detection_indices])
            loca_emb          = np.array([dets[i].detection_data['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i].detection_data['pose'] for i in detection_indices])
            uv_maps           = np.array([dets[i].detection_data['uv'] for i in detection_indices])
            targets           = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix       = self.metric.distance([appe_emb, loca_emb, pose_emb, uv_maps], targets, dims=[self.A_dim, self.P_dim, self.L_dim], phalp_tracker=self.phalp_tracker)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]

        self.metric.matching_threshold = 150 # FIXME
        print(f"Matching threshold: {self.metric.matching_threshold}")
        # exit()
        
        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)


        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d.detection_data['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        if(self.cfg.use_gt): 
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if(t_gt==d_gt): matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks     = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]
            return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]

    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.cfg, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection.detection_data, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1

    def accumulate_vectors(self, track_ids, features="APL"):

        print(track_ids)
        
        a_features = []; p_features = []; l_features = []; t_features = []; l_time     = []; confidence = []; is_tracks  = 0; p_data = []
        for track_idx in track_ids:
            t_features.append([self.tracks[track_idx].track_data['history'][i]['time'] for i in range(self.cfg.phalp.track_history)])
            l_time.append(self.tracks[track_idx].time_since_update)
                
            if("L" in features):  l_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['loca'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  p_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['pose'] for i in range(self.cfg.phalp.track_history)]))
            if("P" in features):  t_id = self.tracks[track_idx].track_id; p_data.append([[data['xy'][0], data['xy'][1], data['scale'], data['scale'], data['time'], t_id] for data in self.tracks[track_idx].track_data['history']])
            if("L" in features):  confidence.append(np.array([self.tracks[track_idx].track_data['history'][i]['conf'] for i in range(self.cfg.phalp.track_history)]))
            is_tracks = 1

        l_time         = np.array(l_time)
        t_features     = np.array(t_features)
        if("P" in features): p_features     = np.array(p_features)
        if("P" in features): p_data         = np.array(p_data)
        if("L" in features): l_features     = np.array(l_features)
        if("L" in features): confidence     = np.array(confidence)
        
        if(is_tracks):
            with torch.no_grad():
                if("P" in features): p_pred, ava_pred, smpl_pred = self.phalp_tracker.forward_for_tracking([p_features, p_data, t_features], "P", l_time)
                if("L" in features): l_pred = self.phalp_tracker.forward_for_tracking([l_features, t_features, confidence], "L", l_time)    
                
            for p_id, track_idx in enumerate(track_ids):
                self.tracks[track_idx].add_predicted(pose=p_pred[p_id] if("P" in features) else None,
                                                     loca=l_pred[p_id] if("L" in features) else None)
                
                self.tracks[track_idx].track_data['prediction']['ava'] = ava_pred[p_id] if("P" in features) else None
                for key in smpl_pred.keys():
                    self.tracks[track_idx].track_data['prediction'][f'prediction_smpl_{key}'] = []
                for key in smpl_pred.keys():
                    print(np.array(smpl_pred[key][p_id]).shape, f"{key}")
                    self.tracks[track_idx].track_data['prediction'][f'prediction_smpl_{key}'].append(smpl_pred[key][p_id] if("P" in features) else None)



class my_lart_transformer(nn.Module):
    def __init__(self, opt, phalp_cfg, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., droppath = 0., device=None):
        super().__init__()
        self.cfg  = opt
        self.phalp_cfg = phalp_cfg
        self.dim  = dim
        self.mask_token = nn.Parameter(torch.randn(self.dim,))
        self.class_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        self.pos_embedding = nn.Parameter(positionalencoding1d(self.dim, 10000))
        self.pos_embedding_learned1 = nn.Parameter(torch.randn(1, self.cfg.frame_length, self.dim))
        self.pos_embedding_learned2 = nn.Parameter(torch.randn(1, self.cfg.frame_length, self.dim))
        self.register_buffer('pe', self.pos_embedding)
        
        self.transformer    = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)
        self.transformer1       = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, drop_path = droppath)
        self.transformer2       = Transformer(self.dim, 1, heads, dim_head, mlp_dim, dropout, drop_path = droppath)

        pad                 = self.cfg.transformer.conv.pad
        stride              = self.cfg.transformer.conv.stride
        kernel              = stride + 2 * pad
        self.conv_en        = nn.Conv1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)
        self.conv_de        = nn.ConvTranspose1d(self.dim, self.dim, kernel_size=kernel, stride=stride, padding=pad)

        # Pose shape encoder for encoding pose shape features, used by default
        self.pose_shape_encoder     = nn.Sequential(
                                            nn.Linear(self.cfg.extra_feat.pose_shape.dim, self.cfg.extra_feat.pose_shape.mid_dim), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.extra_feat.pose_shape.mid_dim, self.cfg.extra_feat.pose_shape.en_dim),
                                        )
        
        # SMPL head for predicting SMPL parameters
        self.smpl_head              = nn.ModuleList([SMPLHead(self.phalp_cfg, input_dim=self.cfg.in_feat, pool='pooled') for _ in range(self.cfg.num_smpl_heads)])
        
        # Location head for predicting 3D location of the person
        self.loca_head              = nn.ModuleList([nn.Sequential(
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat), 
                                            nn.ReLU(), 
                                            nn.Linear(self.cfg.in_feat, self.cfg.in_feat),
                                            nn.ReLU(),         
                                            nn.Linear(self.cfg.in_feat, 3)
                                        ) for _ in range(self.cfg.num_smpl_heads)])
        
        # Action head for predicting action class in AVA dataset labels  
        ava_action_classes          = self.cfg.ava.num_action_classes if not(self.cfg.ava.predict_valid) else self.cfg.ava.num_valid_action_classes
        self.action_head_ava        = nn.ModuleList([nn.Sequential(    
                                            nn.Linear(self.cfg.in_feat, ava_action_classes),
                                        ) for _ in range(self.cfg.num_smpl_heads)])

    def bert_mask(self, data, mask_type):
        if(mask_type=="random"):
            has_detection  = data['has_detection']==1
            mask_detection = data['mask_detection']
            for i in range(data['has_detection'].shape[0]):
                indexes        = has_detection[i].nonzero()
                indexes_mask   = indexes[torch.randperm(indexes.shape[0])[:int(indexes.shape[0]*self.cfg.mask_ratio)]]
                mask_detection[i, indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2]] = 1.0

        elif(mask_type=="zero"):
            has_detection  = data['has_detection']==0
            mask_detection = data['mask_detection']
            indexes_mask   = has_detection.nonzero()
            mask_detection[indexes_mask[:, 0], indexes_mask[:, 1], indexes_mask[:, 2], :] = 1.0
            has_detection = has_detection*0 + 1.0
        
        else:
            raise NotImplementedError
            
        return data, has_detection, mask_detection

    def forward(self, data, mask_type="random"):
        
        # prepare the input data and masking
        data, has_detection, mask_detection = self.bert_mask(data, mask_type)

        # encode the input pose tokens
        pose_   = data['pose_shape'].float()
        pose_en = self.pose_shape_encoder(pose_)
        x       = pose_en
        
        # mask the input tokens
        x[mask_detection[:, :, :, 0]==1] = self.mask_token

        BS, T, P, dim = x.size()
        x = x.view(BS, T*P, dim)

        # adding 2D posistion embedding
        # x = x + self.pos_embedding[None, :, :self.cfg.frame_length, :self.cfg.max_people].reshape(1, dim, self.cfg.frame_length*self.cfg.max_people).permute(0, 2, 1)
        
        x    = x + self.pos_embedding_learned1
        x    = self.transformer1(x, [has_detection, mask_detection])

        x = x.transpose(1, 2)
        x = self.conv_en(x)
        x = self.conv_de(x)
        x = x.transpose(1, 2)
        x = x.contiguous()

        x                = x + self.pos_embedding_learned2
        has_detection    = has_detection*0 + 1
        mask_detection   = mask_detection*0
        x    = self.transformer2(x, [has_detection, mask_detection])
        x = torch.concat([self.class_token.repeat(BS, self.cfg.max_people, 1), x], dim=1)
        

        return x, 0


class My_Pose_transformer_v2(nn.Module):
    
    def __init__(self, cfg, phalp_tracker):
        super(My_Pose_transformer_v2, self).__init__()
        
        self.phalp_cfg = cfg

        # load a config file
        self.cfg = OmegaConf.load(self.phalp_cfg.pose_predictor.config_path).configs
        self.cfg.max_people = 1
        self.encoder      = my_lart_transformer(   
                                opt         = self.cfg, 
                                phalp_cfg   = self.phalp_cfg,
                                dim         = self.cfg.in_feat,
                                depth       = self.cfg.transformer.depth,
                                heads       = self.cfg.transformer.heads,
                                mlp_dim     = self.cfg.transformer.mlp_dim,
                                dim_head    = self.cfg.transformer.dim_head,
                                dropout     = self.cfg.transformer.dropout,
                                emb_dropout = self.cfg.transformer.emb_dropout,
                                droppath    = self.cfg.transformer.droppath,
                                )
        
        self.mean_, self.std_ = np.load(self.phalp_cfg.pose_predictor.mean_std, allow_pickle=True)
        self.mean_            = np.concatenate((self.mean_, np.zeros((1, 229-self.mean_.shape[1]))), axis=1)
        self.std_             = np.concatenate((self.std_, np.ones((1, 229-self.std_.shape[1]))), axis=1)
        self.mean_, self.std_ = torch.tensor(self.mean_), torch.tensor(self.std_)
        self.mean_, self.std_ = self.mean_.float(), self.std_.float()
        self.mean_, self.std_ = self.mean_.unsqueeze(0), self.std_.unsqueeze(0)   
        self.register_buffer('mean', self.mean_)
        self.register_buffer('std', self.std_)
        
        self.smpl = phalp_tracker.HMAR.smpl
            
    def load_weights(self, path):
        # import ipdb; ipdb.set_trace()
        checkpoint_file = torch.load(path)
        # checkpoint_file_filtered = {k[8:]: v for k, v in checkpoint_file['state_dict'].items()} # remove "encoder." from keys
        checkpoint_file_filtered = {k.replace("encoder.", ""): v for k, v in checkpoint_file['state_dict'].items()} # remove "encoder." from keys
        out = self.encoder.load_state_dict(checkpoint_file_filtered, strict=False)
    def readout_pose(self, output):
        
        # return predicted gt pose, betas and location
        BS = output.shape[0]
        FL = output.shape[1]
        pose_tokens      = output.contiguous()
        pose_tokens_     = rearrange(pose_tokens, 'b tp dim -> (b tp) dim')
        
        pred_smpl_params = [self.encoder.smpl_head[i](pose_tokens_)[0] for i in range(self.cfg.num_smpl_heads)]
        pred_cam         = [self.encoder.loca_head[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)]
        pred_ava         = [self.encoder.action_head_ava[i](pose_tokens) for i in range(self.cfg.num_smpl_heads)]
        
        pred_cam         = torch.stack(pred_cam, dim=0)[0]
        pred_cam         = rearrange(pred_cam, 'b (t p) dim -> b t p dim', b=BS, t=FL ,p=self.cfg.max_people) # (BS, T, P, 3)        
        

        global_orient    = rearrange(pred_smpl_params[0]['global_orient'], '(b t p) x y z -> b t p (x y z)', b=BS, t=FL ,p=self.cfg.max_people, x=1, y=3, z=3) # (BS, T, P, 9)
        body_pose        = rearrange(pred_smpl_params[0]['body_pose'], '(b t p) x y z -> b t p (x y z)', b=BS, t=FL ,p=self.cfg.max_people, x=23, y=3, z=3) # (BS, T, P, 207)
        betas            = rearrange(pred_smpl_params[0]['betas'], '(b t p) z -> b t p z', b=BS, t=FL ,p=self.cfg.max_people, z=10) # (BS, T, P, 10)
        pose_vector      = torch.cat((global_orient, body_pose, betas, pred_cam), dim=-1) # (BS, T, P, 229)
        
        pred_ava         = torch.stack(pred_ava, dim=0)[0]
        pred_ava         = rearrange(pred_ava, 'b (t p) dim -> b t p dim', b=BS, t=FL ,p=self.cfg.max_people) # (BS, T, P, 60)        

        # TODO: apply moving average for pridictions

        smpl_outputs = {
            'pose_camera'      : pose_vector,
            'camera'           : pred_cam,
            'ava_action'       : pred_ava,
            'temporal_pred_smpl_params' : pred_smpl_params,
            'temporal_pose_tokens'     : pose_tokens_
        }
            
        return smpl_outputs
            
    def predict_next(self, en_pose, en_data, en_time, time_to_predict):
        
        """encoder takes keys : 
                    pose_shape (bs, self.cfg.frame_length, 229)
                    has_detection (bs, self.cfg.frame_length, 1), 1 if there is a detection, 0 otherwise
                    mask_detection (bs, self.cfg.frame_length, 1)*0       
        """
        
        # set number of people to one 
        n_p = 1
        pose_shape_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 229)
        has_detection_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 1)
        mask_detection_ = torch.zeros(en_pose.shape[0], self.cfg.frame_length, n_p, 1)
        
        # loop thorugh each person and construct the input data
        t_end = []
        for p_ in range(en_time.shape[0]):
            t_min = en_time[p_, 0].min()
            # loop through time 
            for t_ in range(en_time.shape[1]):
                # get the time from start.
                t = min(en_time[p_, t_] - t_min, self.cfg.frame_length - 1)
                
                # get the pose
                pose_shape_[p_, t, 0, :] = en_pose[p_, t_, :]
                
                # get the mask
                has_detection_[p_, t, 0, :] = 1
            t_end.append(t.item())
            
        input_data = {
            "pose_shape" : (pose_shape_ - self.mean_[:, :, None, :]) / (self.std_[:, :, None, :] + 1e-10),
            "has_detection" : has_detection_,
            "mask_detection" : mask_detection_
        }
        
        # place all the data in cuda
        input_data = {k: v.cuda() for k, v in input_data.items()}

        # single forward pass
        output, _ = self.encoder(input_data, self.cfg.mask_type_test)
        decoded_output = self.readout_pose(output[:, self.cfg.max_people:, :])
        print(decoded_output.keys())
        
        assert len(t_end) == len(time_to_predict)
        t_end += time_to_predict + 1

        # last_time = en_time[:, -1].item()
        # print(f"last_time: {last_time}, {en_time}")
        
        predicted_pose_camera_at_t = []
        predicted_ava_action_at_t = []
        predicted_smpl_params_at_t = {
            'global_orient' : [],
            'body_pose' : [],
            'betas' : [],
        }
        for i in range(en_time.shape[0]): 
            t_x = min(t_end[i], self.cfg.frame_length-1)
            last_time = en_time[i, -1].item()
            print(f"Looking at track {i}, at time {last_time}, en_time: {en_time}")
            predicted_pose_camera_at_t.append(decoded_output['pose_camera'][:, t_x, 0, :])
            predicted_ava_action_at_t.append(decoded_output['ava_action'][:, t_x, 0, :])
            print(f"Body pose shape: {np.array(decoded_output['temporal_pred_smpl_params'][0]['body_pose'][t_x].detach().cpu()).shape}")
            predicted_smpl_params_at_t['global_orient'].append(decoded_output['temporal_pred_smpl_params'][0]['global_orient'][t_x])
            predicted_smpl_params_at_t['body_pose'].append(decoded_output['temporal_pred_smpl_params'][0]['body_pose'][t_x])
            predicted_smpl_params_at_t['betas'].append(decoded_output['temporal_pred_smpl_params'][0]['betas'][t_x])
        predicted_pose_camera_at_t = torch.stack(predicted_pose_camera_at_t, dim=0)[0]
        predicted_ava_action_at_t = torch.stack(predicted_ava_action_at_t, dim=0)[0]
        predicted_smpl_params_at_t = {k: torch.stack(v, dim=0) for k, v in predicted_smpl_params_at_t.items()}
                    
        return predicted_pose_camera_at_t, predicted_ava_action_at_t, predicted_smpl_params_at_t
    
    def add_slowfast_features(self, fast_track):
        # add slowfast features to the fast track
        from slowfast.utils.parser import load_config, parse_args
        from slowfast.config.defaults import assert_and_infer_cfg
        from slowfast.visualization.predictor import ActionPredictor, Predictor
        from phalp.models.predictor.wrapper_pyslowfast import SlowFastWrapper

        device = 'cuda'
        path_to_config = "/private/home/jathushan/3D/slowfast/configs/AVA/MViT-L-312_masked.yaml"
        center_crop = False
        if("MViT" in path_to_config): 
            center_crop = True

        self.cfg.opts = None
        cfg = load_config(self.cfg, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        video_model    = Predictor(cfg=cfg, gpu_id=None)
        seq_length     = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

        list_of_frames = fast_track['frame_name']
        list_of_bbox   = fast_track['frame_bbox']
        list_of_fids   = fast_track['fid']
        fast_track['mvit_emb'] = []
        fast_track['action_emb'] = []

        NUM_STEPS        = 6 # 5Hz
        NUM_FRAMES       = seq_length
        list_iter        = list(range(len(list_of_frames)//NUM_STEPS + 1))

        for t_, time_stamp in enumerate(list_iter):    

            start_      = time_stamp * NUM_STEPS
            end_        = (time_stamp + 1) * NUM_STEPS if (time_stamp + 1) * NUM_STEPS < len(list_of_frames) else len(list_of_frames)
            time_stamp_ = list_of_frames[start_:end_]
            if(len(time_stamp_)==0): continue

            mid_        = (start_ + end_)//2
            mid_frame   = list_of_frames[mid_]
            mid_bbox    = list_of_bbox[mid_]
            mid_fid     = list_of_fids[mid_]

            list_of_all_frames = []
            for i in range(-NUM_FRAMES//2,NUM_FRAMES//2 + 1):
                if(mid_ + i < 0):
                    frame_id = 0
                elif(mid_ + i >= len(list_of_frames)):
                    frame_id = len(list_of_frames) - 1
                else:
                    frame_id = mid_ + i
                list_of_all_frames.append(list_of_frames[frame_id])


            mid_bbox_   = mid_bbox.reshape(1, 4).astype(np.int32)
            mid_bbox_   = np.concatenate([mid_bbox_[:, :2], mid_bbox_[:, :2] + mid_bbox_[:, 2:4]], 1)
            # img1 = cv2.imread(mid_frame)
            # img1 = cv2.rectangle(img1, (mid_bbox_[0, 0], mid_bbox_[0, 1]), (mid_bbox_[0, 2], mid_bbox_[0, 3]), (0, 255, 0), 2)
            # cv2.imwrite("test.png", img1)
            with torch.no_grad():
                task_      = SlowFastWrapper(t_, cfg, list_of_all_frames, mid_bbox_, video_model, center_crop=center_crop)
                preds      = task_.action_preds[0]
                feats      = task_.action_preds[1]
                preds      = preds.cpu().numpy()
                feats      = feats.cpu().numpy()

            for frame_ in time_stamp_:
                fast_track['mvit_emb'].append(feats)
                fast_track['action_emb'].append(preds)
        
        assert len(fast_track['mvit_emb']) == len(fast_track['frame_name'])
        assert len(fast_track['action_emb']) == len(fast_track['frame_name'])
        fast_track['mvit_emb'] = np.array(fast_track['mvit_emb'])
        fast_track['action_emb'] = np.array(fast_track['action_emb'])
        
        return fast_track

    def smooth_tracks(self, fast_track, moving_window=False, step=1, window=20):
        
        if("mvit" in self.cfg.extra_feat.enable):
            fast_track = self.add_slowfast_features(fast_track)

        # set number of people to one 
        n_p = 1
        fl  = fast_track['pose_shape'].shape[0]

        pose_shape_all = torch.zeros(1, fl, n_p, 229)
        has_detection_all = torch.zeros(1, fl, n_p, 1)
        mask_detection_all = torch.zeros(1, fl, n_p, 1)

        if("mvit" in self.cfg.extra_feat.enable):
            mvit_feat_all = fast_track['mvit_emb'][None, :, :,]
        
        if("joints_3D" in self.cfg.extra_feat.enable):
            joints_ = fast_track['3d_joints'][:, :, :, :]
            camera_ = fast_track['camera'][:, None, :, :]
            joints_3d_all = joints_ + camera_
            joints_3d_all = joints_3d_all.reshape(1, fl, n_p, 135)

        for t_ in range(fast_track['pose_shape'].shape[0]):
            pose_shape_all[0, t_, 0, :] = torch.tensor(fast_track['pose_shape'][t_])
            has_detection_all[0, t_, 0, :] = 1
            mask_detection_all[0, t_, 0, :] = 1.0 - torch.tensor(fast_track['has_detection'][t_, 0])

        S_ = 0
        STEP_ = step
        WINDOW_ = window
        w_steps = range(S_, S_+fl, STEP_)
        assert 2*WINDOW_ + STEP_ < self.cfg.frame_length
        STORE_OUTPUT_ = torch.zeros(1, fl, self.cfg.in_feat)

        for w_ in w_steps:

            pose_shape_ = torch.zeros(1, self.cfg.frame_length, n_p, 229)
            has_detection_ = torch.zeros(1, self.cfg.frame_length, n_p, 1)
            mask_detection_ = torch.zeros(1, self.cfg.frame_length, n_p, 1)

            start_ = w_ - WINDOW_ if (w_ - WINDOW_>0) else 0
            end_ = w_ + STEP_ + WINDOW_ if (w_ + STEP_ + WINDOW_<=fl) else fl

            pose_shape_[:, :end_-start_, :, :] = pose_shape_all[:, start_:end_, :, :]
            has_detection_[:, :end_-start_, :, :] = has_detection_all[:, start_:end_, :, :]
            mask_detection_[:, :end_-start_, :, :] = mask_detection_all[:, start_:end_, :, :]

            input_data = {
                "pose_shape" : (pose_shape_ - self.mean_[0, :, None, :]) / (self.std_[0, :, None, :] + 1e-10),
                "has_detection" : has_detection_,
                "mask_detection" : mask_detection_
            }
            
            # add other features if enables:
            if("joints_3D" in self.cfg.extra_feat.enable):
                joints_ = torch.zeros(1, self.cfg.frame_length, n_p, 135)
                joints_[:, :end_-start_, :, :] = torch.tensor(joints_3d_all[:, start_:end_, :, :])
                input_data["joints_3D"] = joints_

            if("mvit" in self.cfg.extra_feat.enable):
                mvit_ = torch.zeros(1, self.cfg.frame_length, n_p, 1152)
                mvit_[:, :end_-start_, :, :] = torch.tensor(mvit_feat_all[:, start_:end_, :, :])
                input_data["mvit_emb"] = mvit_

            input_data = {k: v.cuda() for k, v in input_data.items()}

            output, _ = self.encoder(input_data, self.cfg.mask_type_test)
            output = output[:, self.cfg.max_people:, :]

            
            if(w_+STEP_<fl):
                if(w_<=WINDOW_):
                    STORE_OUTPUT_[:,  w_:w_+STEP_, :] = output[:,  w_:w_+STEP_, :]
                else:
                    STORE_OUTPUT_[:,  w_:w_+STEP_, :] = output[:,  WINDOW_:WINDOW_+STEP_, :]
            else:
                if(w_<=WINDOW_):
                    STORE_OUTPUT_[:,  w_:fl, :] = output[:,  w_:fl, :]
                else:
                    STORE_OUTPUT_[:,  w_:fl, :] = output[:,  WINDOW_:WINDOW_+(fl-w_), :]

        decoded_output = self.readout_pose(STORE_OUTPUT_.cuda())

        fast_track['pose_shape'] = decoded_output['pose_camera'][0, :fast_track['pose_shape'].shape[0], :, :]
        fast_track['cam_smoothed'] = decoded_output['camera'][0, :fast_track['pose_shape'].shape[0], :, :]
        fast_track['ava_action'] = decoded_output['ava_action'][0, :fast_track['pose_shape'].shape[0], :, :]
        
        return fast_track

def to_jsonable(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().tolist()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.generic, np.bool_)):  # numpy scalars
        return val.item()
    elif isinstance(val, dict):
        return {k: to_jsonable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_jsonable(v) for v in val]
    else:
        return val

class TokenHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        from .tokenhmr.lib.models import load_tokenhmr

        model, _ = load_tokenhmr(
            checkpoint_path=cfg.checkpoint,
            model_cfg=cfg.model_config,
            is_train_state=False,
            is_demo=True
        )
        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:, :3, :, :],
            'mask': (x[:, 3, :, :]).clip(0, 1),
        }
        model_out = self.model(batch)

        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out


class PHALP_Prime_TokenHMR(PHALP):
    def __init__(self, cfg, video_cfg):
        super().__init__(cfg)
        self.video_cfg = video_cfg
        self.io_manager.output_fps = self.video_cfg["fps"] if "fps" in self.video_cfg else 24

    def setup_hmr(self):
        self.HMAR = TokenHMRPredictor(self.cfg)

    def setup_predictor(self):
        self.pose_predictor = My_Pose_transformer_v2(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)

    def setup_deepsort(self):
        # log.info("Setting up DeepSort...")
        metric  = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        # self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])  
        self.tracker = My_Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])  

    def get_human_features(self, image, seg_mask, bbox, bbox_pad, score, frame_name, cls_id, t_, measurments, gt=1, ann=None, extra_data=None):
        NPEOPLE = len(score)

        if len(seg_mask) > 0:
            # save the image with the mask overlayed
            mask_uint8 = (seg_mask[0].astype(np.uint8)) * 255
            mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
            image_overlay = cv2.addWeighted(image, 0.3, mask_rgb, 0.7, 0)
            # Replace the path to point to MASKS directory
            mask_save_path = frame_name.replace("_DEMO/video/img/", "MASKS/")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            # Save the overlaid image
            cv2.imwrite(mask_save_path, image_overlay)

            # also save the mask as numpy array
            np.save(mask_save_path.replace(".png", ".npy"), seg_mask[0])


        if(NPEOPLE==0): return []

        img_height, img_width, new_image_size, left, top = measurments                
        ratio = 1.0/int(new_image_size)*self.cfg.render.res
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []
        for p_ in range(NPEOPLE):
            if bbox[p_][2]-bbox[p_][0]<self.cfg.phalp.small_w or bbox[p_][3]-bbox[p_][1]<self.cfg.phalp.small_h:
                continue
            masked_image, center_, scale_, rles, center_pad, scale_pad = self.get_croped_image(image, bbox[p_], bbox_pad[p_], seg_mask[p_])
            masked_image_list.append(masked_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            rles_list.append(rles)
            selected_ids.append(p_)
        
        if(len(masked_image_list)==0): return []

        masked_image_list = torch.stack(masked_image_list, dim=0)
        BS = masked_image_list.size(0)
        
        with torch.no_grad():
            extra_args      = {}
            hmar_out        = self.HMAR(masked_image_list.cuda(), **extra_args)
            uv_vector       = hmar_out['uv_vector']
            appe_embedding  = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding  = appe_embedding.view(appe_embedding.shape[0], -1)
            
            # save the appe_embedding as numpy array
            os.makedirs(self.cfg.video.output_dir + "/APPE_EMBEDDING/", exist_ok=True)
            np.save(self.cfg.video.output_dir + "/APPE_EMBEDDING/" + frame_name.split("/")[-1].replace(".jpg", ".npy"), appe_embedding.cpu().numpy())

            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(hmar_out['pose_smpl'], hmar_out['pred_cam'],
                                                                                               center=(np.array(center_list) + np.array([left, top]))*ratio,
                                                                                               img_size=self.cfg.render.res,
                                                                                               scale=np.max(np.array(scale_list), axis=1, keepdims=True)*ratio)
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]
            
            if(self.cfg.phalp.pose_distance=="joints"):
                pose_embedding  = pred_joints.cpu().view(BS, -1)
            elif(self.cfg.phalp.pose_distance=="smpl"):
                pose_embedding = []
                for i in range(BS):
                    pose_embedding_  = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            pred_joints_2d_ = pred_joints_2d.reshape(BS,-1)/self.cfg.render.res
            pred_cam_ = pred_cam.view(BS, -1)
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()
            loca_embedding  = torch.cat((pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 1)

        # save the pose_embedding as numpy array
        os.makedirs(self.cfg.video.output_dir + "/POSE_EMBEDDING/", exist_ok=True)
        np.save(self.cfg.video.output_dir + "/POSE_EMBEDDING/" + frame_name.split("/")[-1].replace(".jpg", ".npy"), pose_embedding.cpu().numpy())

        # save the loca_embedding as numpy array
        os.makedirs(self.cfg.video.output_dir + "/LOCA_EMBEDDING/", exist_ok=True)
        np.save(self.cfg.video.output_dir + "/LOCA_EMBEDDING/" + frame_name.split("/")[-1].replace(".jpg", ".npy"), loca_embedding.cpu().numpy())
        
        # keeping it here for legacy reasons (T3DP), but it is not used.
        full_embedding    = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding.cpu()), 1)
        
        detection_data_list = []
        detection_data_list_save = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                                "bbox"            : np.array([bbox[p_][0], bbox[p_][1], (bbox[p_][2] - bbox[p_][0]), (bbox[p_][3] - bbox[p_][1])]),
                                "mask"            : rles_list[i],
                                "conf"            : score[p_], 
                                
                                "appe"            : appe_embedding[i].cpu().numpy(), 
                                "pose"            : pose_embedding[i].numpy(), 
                                "loca"            : loca_embedding[i].cpu().numpy(), 
                                "uv"              : uv_vector[i].cpu().numpy(), 
                                
                                "embedding"       : full_embedding[i], 
                                "center"          : center_list[i],
                                "scale"           : scale_list[i],
                                "smpl"            : pred_smpl_params[i],
                                "camera"          : pred_cam_[i].cpu().numpy(),
                                "camera_bbox"     : hmar_out['pred_cam'][i].cpu().numpy(),
                                "3d_joints"       : pred_joints[i].cpu().numpy(),
                                "2d_joints"       : pred_joints_2d_[i].cpu().numpy(),
                                "size"            : [img_height, img_width],
                                "img_path"        : frame_name,
                                "img_name"        : frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                                "class_name"      : cls_id[p_],
                                "time"            : t_,

                                "ground_truth"    : gt[p_],
                                "annotations"     : ann[p_],
                                "extra_data"      : extra_data[p_] if extra_data is not None else None
                            }
            detection_data_list.append(Detection(detection_data))
            # Convert all NumPy objects to JSON-compatible types
            detection_data_jsonable = {k: to_jsonable(v) for k, v in detection_data.items()}
            detection_data_list_save.append(detection_data_jsonable)

        # save dict as json for each frame
        os.makedirs(self.cfg.video.output_dir + "/DETECTION_DATA/", exist_ok=True)
        for i, detection_data in enumerate(detection_data_list_save):
            with open(self.cfg.video.output_dir + "/DETECTION_DATA/" + frame_name.split("/")[-1].replace(".jpg", f"_{i}.json"), "w") as f:
                json.dump(detection_data, f)

        return detection_data_list

    def track_with_info(self):
        
        eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
        history_keys    = ['appe', 'loca', 'pose', 'uv'] if self.cfg.render.enable else []
        prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca', 'prediction_ava', 'prediction_smpl_global_orient', 'prediction_smpl_body_pose', 'prediction_smpl_betas'] if self.cfg.render.enable else []
        extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'class_name', 'conf', 'annotations']
        extra_keys_2    = ['smpl', 'camera', 'camera_bbox', '3d_joints', '2d_joints', 'mask', 'extra_data']
        history_keys    = history_keys + extra_keys_1 + extra_keys_2
        visual_store_   = eval_keys + history_keys + prediction_keys
        tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
        
        # process the source video and return a list of frames
        # source can be a video file, a youtube link or a image folder
        io_data = self.io_manager.get_frames_from_source()
        list_of_frames, additional_data = io_data['list_of_frames'], io_data['additional_data']
        self.cfg.video_seq = io_data['video_name']
        pkl_path = self.cfg.video.output_dir + '/results/' + self.cfg.track_dataset + "_" + str(self.cfg.video_seq) + '.pkl'
        video_path = self.cfg.video.output_dir + '/' + self.cfg.base_tracker + '_' + str(self.cfg.video_seq) + '.mp4'
        
        # check if the video is already processed                                  
        if(not(self.cfg.overwrite) and os.path.isfile(pkl_path)): 
            return 0
        
        # eval mode
        self.eval()
        
        # setup rendering, deep sort and directory structure
        self.setup_deepsort()
        self.default_setup()
        
        log.info("Saving tracks at : " + self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq))
        
        # try: 
            
        list_of_frames = list_of_frames if self.cfg.phalp.start_frame==-1 else list_of_frames[self.cfg.phalp.start_frame:self.cfg.phalp.end_frame]
        list_of_shots = self.get_list_of_shots(list_of_frames)
        
        tracked_frames = []
        final_visuals_dic = {}
        
        for t_, frame_name in progress_bar(enumerate(list_of_frames), description="Tracking : " + self.cfg.video_seq, total=len(list_of_frames), disable=False):
            
            image_frame               = self.io_manager.read_frame(frame_name)
            img_height, img_width, _  = image_frame.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            self.cfg.phalp.shot       = 1 if t_ in list_of_shots else 0

            if(self.cfg.render.enable):
                # reset the renderer
                # TODO: add a flag for full resolution rendering
                self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
                self.visualizer.reset_render(self.cfg.render.res*self.cfg.render.up_scale)
            
            ############ detection ##############
            pred_bbox, pred_bbox_pad, pred_masks, pred_scores, pred_classes, gt_tids, gt_annots = self.get_detections(image_frame, frame_name, t_, additional_data, measurments)

            ############ Run EXTRA models to attach to the detections ##############
            extra_data = self.run_additional_models(image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots)
            
            ############ HMAR ##############
            detections = self.get_human_features(image_frame, pred_masks, pred_bbox, pred_bbox_pad, pred_scores, frame_name, pred_classes, t_, measurments, gt_tids, gt_annots, extra_data)

            ############ tracking ##############
            self.tracker.predict()
            print(frame_name)
            self.tracker.update(detections, t_, frame_name, self.cfg.phalp.shot)

            ############ record the results ##############
            final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': self.cfg.phalp.shot, 'frame_path': frame_name})
            if(self.cfg.render.enable): final_visuals_dic[frame_name]['frame'] = image_frame
            for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []

            print(len( self.tracker.tracks))
            
            ############ record the track states (history and predictions) ##############
            for tracks_ in self.tracker.tracks:
                if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                if(not(tracks_.is_confirmed())): continue
                
                track_id        = tracks_.track_id
                track_data_hist = tracks_.track_data['history'][-1]
                track_data_pred = tracks_.track_data['prediction']

                # print(track_data_hist.keys())
                # print(track_data_pred.keys())
                # exit()

                final_visuals_dic[frame_name]['tid'].append(track_id)
                final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                for hkey_ in history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                # for pkey_ in prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])
                for pkey_ in prediction_keys: 
                    # print(pkey_)
                    if "smpl" in pkey_:
                        final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_][-1])
                    else:
                        final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])

                if(tracks_.time_since_update==0):
                    final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                    final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                    
                    if(tracks_.hits==self.cfg.phalp.n_init):
                        for pt in range(self.cfg.phalp.n_init-1):
                            track_data_hist_ = tracks_.track_data['history'][-2-pt]
                            track_data_pred_ = tracks_.track_data['prediction']
                            frame_name_      = tracked_frames[-2-pt]
                            final_visuals_dic[frame_name_]['tid'].append(track_id)
                            final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                            final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_time'].append(0)

                            for hkey_ in history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                            # for pkey_ in prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])
                            for pkey_ in prediction_keys:
                                if "smpl" in pkey_:
                                    final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_][-1])
                                else:
                                    final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])

            if(tracks_.hits==self.cfg.phalp.n_init):
                print(final_visuals_dic[frame_name].keys())
                for key_ in final_visuals_dic[frame_name].keys():
                    print(key_, np.array(final_visuals_dic[frame_name][key_]).shape)
                # print(np.array(final_visuals_dic[frame_name]['prediction_smpl_body_pose']).shape)  
                exit()     
    
            ############ save the video ##############
            if(self.cfg.render.enable and t_>=self.cfg.phalp.n_init):                    
                d_ = self.cfg.phalp.n_init+1 if(t_+1==len(list_of_frames)) else 1
                for t__ in range(t_, t_+d_):

                    frame_key = list_of_frames[t__-self.cfg.phalp.n_init]
                    rendered_, f_size = self.visualizer.render_video(final_visuals_dic[frame_key])      

                    # save the rendered frame
                    self.io_manager.save_video(video_path, rendered_, f_size, t=t__-self.cfg.phalp.n_init)
                    # delete the frame after rendering it
                    del final_visuals_dic[frame_key]['frame']
                    
                    # # delete unnecessary keys
                    # for tkey_ in tmp_keys_:  
                    #     del final_visuals_dic[frame_key][tkey_] 

        print(final_visuals_dic[list_of_frames[0]].keys())
        exit()

        joblib.dump(final_visuals_dic, pkl_path, compress=3)
        print(f"Saved to {pkl_path}")
        self.io_manager.close_video()
        if(self.cfg.use_gt): joblib.dump(self.tracker.tracked_cost, self.cfg.video.output_dir + '/results/' + str(self.cfg.video_seq) + '_' + str(self.cfg.phalp.start_frame) + '_distance.pkl')


        reader = imageio.get_reader(video_path)
        frames = [frame.astype(np.float32) / 255.0 for frame in reader] 
        reader.close()

        # Save again using diffusers
        export_to_video(frames, output_video_path=video_path, fps=self.io_manager.output_fps)
        
        return final_visuals_dic, pkl_path

        # except Exception as e: 
        #     print(e)
        #     print(traceback.format_exc()) 


@dataclass
class Human4DConfig(FullConfig):
    checkpoint: Optional[str] = None
    model_config: Optional[str] = None
    output_dir: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


class TokenHMRTrackGenerator:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_config: Optional[str] = None,
        overrides: Optional[dict] = None,
    ):
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.overrides = overrides or {}
        
        # Initialize Hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="tokenhmr/lib/configs_hydra")

    def run(self, video_path, output_dir = "outputs_DEL"):
        # Initialize hydra config with overrides
        cfg: DictConfig = hydra.compose(
            config_name="config",
            overrides=[
                f"video.source={video_path}",
                f"video.output_dir={output_dir}",
                *(f"{k}={v}" for k, v in self.overrides.items())
            ]
        )

        video_cfg = os.path.join(os.path.dirname(video_path), "metadata.json")
        with open(video_cfg, "r") as f:
            video_cfg = json.load(f)

        if self.checkpoint:
            cfg.checkpoint = self.checkpoint
        if self.model_config:
            cfg.model_config = self.model_config

        tracker = PHALP_Prime_TokenHMR(cfg, video_cfg)
        # tracker.track()
        tracker.track_with_info()


# Usage example
if __name__ == "__main__":
    runner = TokenHMRTrackGenerator(
        checkpoint="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt",
        model_config="/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml",
        overrides={"render.colors": "slahmr"}
    )
    runner.run("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_JumpingJack_g20_c01/v_JumpingJack_g20_c01_full.mp4")
