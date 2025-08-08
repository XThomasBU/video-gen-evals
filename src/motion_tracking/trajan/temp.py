from __future__ import annotations

import dataclasses
import einops
import os
import io
import jax
import matplotlib.pyplot as plt
import numpy as np
# from tapnet.torch import tapir_model
from tapnet.models import tapir_model
from tapnet.utils import transforms, viz_utils, model_utils
from tapnet.trajan import track_autoencoder
# from tapnet.tapvid import evaluation_datasets
import torch
import torch.nn.functional as F
import mediapy as media

FPS = 25


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
    """(Unmodified) Computes TAP-Vid metrics..."""
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)
    metrics = {}
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == 'first':
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == 'strided':
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError('Unknown query mode ' + query_mode)
    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=summing_axis,
    ) / np.sum(evaluation_points, axis=summing_axis)
    metrics['occlusion_accuracy'] = occ_acc
    all_frac_within = []
    all_jaccard = []
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum((pred_tracks - gt_tracks)**2, axis=-1) < thresh**2
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(is_correct & evaluation_points, axis=summing_axis)
        count_visible = np.sum(visible & evaluation_points, axis=summing_axis)
        metrics[f'pts_within_{thresh}'] = count_correct / count_visible
        all_frac_within.append(metrics[f'pts_within_{thresh}'])
        true_pos = np.sum(is_correct & pred_visible & evaluation_points, axis=summing_axis)
        gt_pos = np.sum(visible & evaluation_points, axis=summing_axis)
        false_pos = np.sum(((~visible) | (~within_dist)) & pred_visible & evaluation_points, axis=summing_axis)
        metrics[f'jaccard_{thresh}'] = true_pos / (gt_pos + false_pos)
        all_jaccard.append(metrics[f'jaccard_{thresh}'])
    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    return metrics


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ProcessTracksForTrackAutoencoder:
    # (Unmodified) existing dataclass
    num_support_tracks: int
    num_target_tracks: int
    before_boundary: bool = True
    episode_length: int = 150
    video_key: str = "video"
    tracks: str = "tracks"
    visible_key: str = "visible"

    def random_map(self, features):
        # (Unmodified) existing random_map logic...
        ...
        return features

def npload(fname):
  loaded = np.load(fname, allow_pickle=False)
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)

def recover_tree(flat_dict):
  tree = (
      {}
  )  # Initialize an empty dictionary to store the resulting tree structure
  for (
      k,
      v,
  ) in (
      flat_dict.items()
  ):  # Iterate over each key-value pair in the flat dictionary
    parts = k.split(
        '/'
    )  # Split the key into parts using "/" as a delimiter to build the tree structure
    node = tree  # Start at the root of the tree
    for part in parts[
        :-1
    ]:  # Loop through each part of the key, except the last one
      if (
          part not in node
      ):  # If the current part doesn't exist as a key in the node, create an empty dictionary for it
        node[part] = {}
      node = node[part]  # Move down the tree to the next level
    node[parts[-1]] = v  # Set the value at the final part of the key
  return tree  # Return the reconstructed tree


class VideoEvaluator:
    """Wraps TAPIR + TrackAutoencoder to compute TAP-Vid scores for any video."""
    def __init__(self):
        # TAPIR setup (exact same)
        MODEL_TYPE = 'bootstapir'
        device = torch.device('cuda')
        checkpoint_path = 'tapnet/checkpoints/bootstapir_checkpoint_v2.npy'
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state['params'], ckpt_state['state']
        kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
        if MODEL_TYPE == 'bootstapir':
            kwargs.update(dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0))
        self.tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)
        # TrackAutoencoder setup (exact same)
        trajan_checkpoint_path = 'tapnet/checkpoints/track_autoencoder_ckpt.npz'
        flat = npload(trajan_checkpoint_path)
        params_tree = recover_tree(flat)
        self.autoencoder = track_autoencoder.TrackAutoEncoder(decoder_scan_chunk_size=32)
        self.ae_params = params_tree['params']
        self.preprocessor = ProcessTracksForTrackAutoencoder(num_support_tracks=2048, num_target_tracks=2048)

    def _read_video(self, path: str):
        video = media.read_video(path)
        frames = media.resize_video(video, (256, 256))
        return video, frames

    def _tapir_inference(self, frames: np.ndarray):
        # (Unmodified) original TAPIR inference logic
        frames_proc = model_utils.preprocess_frames(frames[None])
        feature_grids = self.tapir.get_feature_grids(frames_proc, is_training=False)
        # sample points
        b, t, h, w, _ = frames_proc.shape
        num_points = 4096
        y = np.random.randint(0, h, (num_points, 1))
        x = np.random.randint(0, w, (num_points, 1))
        t_idx = np.random.randint(0, t, (num_points, 1))
        query_points = np.concatenate((t_idx, y, x), axis=-1).astype(np.float32)[None]
        outputs = self.tapir(
            video=frames_proc,
            query_points=query_points,
            is_training=False,
            query_chunk_size=32,
            feature_grids=feature_grids,
        )
        print(outputs.keys())
        exit()
        tracks = np.array(outputs['tracks'][0])
        occlusions = outputs['occlusion'][0]
        expected_dist = outputs['expected_dist'][0]
        visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
        return tracks, visibles

    def evaluate(self, video_path: str) -> dict:
        # Read and predict
        orig_video, frames = self._read_video(video_path)
        tracks, visibles = self._tapir_inference(frames)
        # Prepare autoencoder batch
        tracks_grid = transforms.convert_grid_coordinates(tracks, (256, 256), (1, 1))
        batch = {
            'video': orig_video,
            'tracks': einops.rearrange(tracks_grid + 0.5, 'q t c -> t q c'),
            'visible': einops.rearrange(visibles, 'q t -> t q')
        }
        batch = self.preprocessor.random_map(batch)
        batch.pop('tracks', None)
        # Autoencoder forward
        outputs = self.autoencoder.apply({'params': self.ae_params}, batch)
        rec_tracks = outputs.tracks[0]
        rec_visibles = model_utils.postprocess_occlusions(outputs.visible_logits, outputs.certain_logits)
        # Compute metrics
        metrics = compute_tapvid_metrics(
            query_points=batch['query_points'],
            gt_occluded=1 - batch['target_tracks_visible'][..., :, 0],
            gt_tracks=batch['target_points'][None, ...],
            pred_occluded=rec_visibles[..., :, 0],
            pred_tracks=rec_tracks[None, ...],
            query_mode='strided',
            get_trackwise_metrics=False,
        )
        avg_jaccard = np.mean([metrics[f'jaccard_{d}'] for d in [1, 2, 4, 8, 16]])
        occl_acc = metrics['occlusion_accuracy'].mean()
        return {'average_jaccard': float(avg_jaccard), 'occlusion_accuracy': float(occl_acc)}

if __name__ == "__main__":
    video_path = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/real_videos/BodyWeightSquats/v_BodyWeightSquats_g09_c07/video.mp4"
    evaluator = VideoEvaluator()
    metrics = evaluator.evaluate(video_path)
    print(metrics)