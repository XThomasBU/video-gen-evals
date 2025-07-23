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
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tapnet.utils import model_utils
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
  """Computes TAP-Vid metrics (Jaccard, Pts.

  Within Thresh, Occ.

  Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.
     get_trackwise_metrics: if True, the metrics will be computed for every
       track (rather than every video, which is the default).  This means
       every output tensor will have an extra axis [batch, num_tracks] rather
       than simply (batch).

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  summing_axis = (2,) if get_trackwise_metrics else (1, 2)

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=summing_axis,
  ) / np.sum(evaluation_points, axis=summing_axis)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=summing_axis,
    )
    count_visible_points = np.sum(
        visible & evaluation_points, axis=summing_axis
    )
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=summing_axis
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(
        false_positives & evaluation_points, axis=summing_axis
    )
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics


MODEL_TYPE = 'bootstapir'
device = torch.device('cuda')

checkpoint_path = 'tapnet/checkpoints/bootstapir_checkpoint_v2.npy'

ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == 'bootstapir':
  kwargs.update(
      dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0)
  )
tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

print(tapir)

trajan_checkpoint_path = 'tapnet/checkpoints/track_autoencoder_ckpt.npz'
video = media.read_video("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/HandstandPushups/v_HandStandPushups_g02_c02.mp4")

print(video.shape)




def inference(frames, query_points):
  """Inference on one video.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

  Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
  """
  # Preprocess video to match model inputs format
  frames = model_utils.preprocess_frames(frames)
  query_points = query_points.astype(np.float32)
  frames, query_points = frames[None], query_points[None]  # Add batch dimension

  outputs = tapir(
      video=frames,
      query_points=query_points,
      is_training=False,
      query_chunk_size=32,
  )
  tracks, occlusions, expected_dist = (
      outputs['tracks'],
      outputs['occlusion'],
      outputs['expected_dist'],
  )

  # Binarize occlusions
  visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
  return tracks[0], visibles[0]


inference = jax.jit(inference)


def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(
      np.int32
  )  # [num_point
  return points


resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}
num_points = 4096  # @param {type: "integer"}

frames = media.resize_video(video, (resize_height, resize_width))
frames = model_utils.preprocess_frames(frames[None])
feature_grids = tapir.get_feature_grids(frames, is_training=False)
query_points = sample_random_points(
    frames.shape[1], frames.shape[2], frames.shape[3], num_points
)
chunk_size = 32


def chunk_inference(query_points):
  query_points = query_points.astype(np.float32)[None]

  outputs = tapir(
      video=frames,
      query_points=query_points,
      is_training=False,
      query_chunk_size=chunk_size,
      feature_grids=feature_grids,
  )
  tracks, occlusions, expected_dist = (
      outputs["tracks"],
      outputs["occlusion"],
      outputs["expected_dist"],
  )

  # Binarize occlusions
  visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
  return tracks[0], visibles[0]


chunk_inference = jax.jit(chunk_inference)

all_tracks = []
all_visibles = []
for chunk in range(0, query_points.shape[0], chunk_size):
  tracks, visibles = chunk_inference(query_points[chunk : chunk + chunk_size])
  all_tracks.append(np.array(tracks))
  all_visibles.append(np.array(visibles))

tracks = np.concatenate(all_tracks, axis=0)
visibles = np.concatenate(all_visibles, axis=0)

# Visualize sparse point tracks
height, width = video.shape[1:3]
tracks = transforms.convert_grid_coordinates(
    tracks, (resize_width, resize_height), (width, height)
)
video_viz = viz_utils.paint_point_track(video, tracks, visibles)
media.show_video(video_viz, fps=FPS)


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

params = recover_tree(npload(trajan_checkpoint_path))


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ProcessTracksForTrackAutoencoder:
  """Samples tracks and fills out support_tracks, query_points etc.

  TrackAutoencoder format which will be output from this transform:
   video: float["*B T H W 3"]
   support_tracks: float["*B QS T 2"]
   support_tracks_visible: float["*B QS T 1"]
   query_points: float["*B Q 3"]
  """

  # note that we do not use query points in the encoding, so it is expected
  # that num_support_tracks >> num_target_tracks

  num_support_tracks: int
  num_target_tracks: int

  # If true, assume that everything after the boundary_frame is padding,
  # so don't sample any query points after the boundary_frame, and only sample
  # target tracks that have at least one visible frame before the boundary.
  before_boundary: bool = True
  episode_length: int = 150

  # Keys.
  video_key: str = "video"
  tracks: str = "tracks"  # [time, num_points, 2]
  visible_key: str = "visible"  # [time, num_points, 1]

  def random_map(self, features):

    # set tracks xy and compute visibility
    tracks_xy = features[self.tracks][..., :2]
    tracks_xy = np.asarray(tracks_xy, np.float32)
    boundary_frame = features["video"].shape[0]

    # visibles already post-processed by compute_point_tracks.py
    visibles = np.asarray(features[self.visible_key], np.float32)

    # pad to 'episode_length' frames
    if self.before_boundary:
      # if input video is longer than episode_length, crop to episode_length
      if self.episode_length - visibles.shape[0] < 0:
        visibles = visibles[: self.episode_length]
        tracks_xy = tracks_xy[: self.episode_length]

      visibles = np.pad(
          visibles,
          [[0, self.episode_length - visibles.shape[0]], [0, 0]],
          constant_values=0.0,
      )
      tracks_xy = np.pad(
          tracks_xy,
          [[0, self.episode_length - tracks_xy.shape[0]], [0, 0], [0, 0]],
          constant_values=0.0,
      )

    # Samples indices for support tracks and target tracks.
    num_input_tracks = tracks_xy.shape[1]
    idx = np.arange(num_input_tracks)
    np.random.shuffle(idx)

    assert (
        num_input_tracks >= self.num_support_tracks + self.num_target_tracks
    ), (
        (
            f"num_input_tracks {num_input_tracks} must be greater than"
            f" num_support_tracks {self.num_support_tracks} + num_target_tracks"
            f" {self.num_target_tracks}"
        ),
    )

    idx_support = idx[-self.num_support_tracks :]
    idx_target = idx[: self.num_target_tracks]

    # Gathers support tracks from `features`.  Features are of shape
    # [time, num_points, 2]
    support_tracks = tracks_xy[..., idx_support, :]
    support_tracks_visible = visibles[..., idx_support]

    # Gathers target tracks from `features`.
    target_tracks = tracks_xy[..., idx_target, :]
    target_tracks_visible = visibles[..., idx_target]

    # transpose to [num_points, time, ...]
    support_tracks = np.transpose(support_tracks, [1, 0, 2])
    support_tracks_visible = np.expand_dims(
        np.transpose(support_tracks_visible, [1, 0]), axis=-1
    )

    # [time, point_id, x/y] -> [point_id, time, x/y]
    target_tracks = np.transpose(target_tracks, [1, 0, 2])
    target_tracks_visible = np.transpose(target_tracks_visible, [1, 0])

    # Sample query points as random visible points
    num_target_tracks = target_tracks_visible.shape[0]
    num_frames = target_tracks_visible.shape[1]
    random_frame = np.zeros(num_target_tracks, dtype=np.int64)

    for i in range(num_target_tracks):
      visible_indices = np.where(target_tracks_visible[i] > 0)[0]
      if len(visible_indices) > 0:
        # Choose a random frame index from the visible ones
        random_frame[i] = np.random.choice(visible_indices)
      else:
        # If no frame is visible for a track, default to frame 0
        # (or handle as appropriate for your use case)
        random_frame[i] = 0

    # Create one-hot encoding based on the randomly selected frame for each track
    idx = np.eye(num_frames, dtype=np.float32)[
        random_frame
    ]  # [num_target_tracks, num_frames]

    # Use the one-hot index to select the coordinates at the chosen frame
    target_queries_xy = np.sum(
        target_tracks * idx[..., np.newaxis], axis=1
    )  # [num_target_tracks, 2]

    # Stack frame index and coordinates: [t, x, y]
    target_queries = np.stack(
        [
            random_frame.astype(np.float32),
            target_queries_xy[..., 0],
            target_queries_xy[..., 1],
        ],
        axis=-1,
    )  # [num_target_tracks, 3]

    # Add channel dimension to target_tracks_visible
    target_tracks_visible = np.expand_dims(target_tracks_visible, axis=-1)

    # Updates `features` to contain these *new* features and add batch dim.
    features_new = {
        "support_tracks": support_tracks[None, :],
        "support_tracks_visible": support_tracks_visible[None, :],
        "query_points": target_queries[None, :],
        "target_points": target_tracks[None, :],
        "boundary_frame": np.array([boundary_frame]),
        "target_tracks_visible": target_tracks_visible[None, :],
    }
    features.update(features_new)
    return features


# Create model and define forward pass.
model = track_autoencoder.TrackAutoEncoder(decoder_scan_chunk_size=32)

@jax.jit
def forward(params, inputs):
  outputs = model.apply({'params': params}, inputs)
  return outputs

# Create preprocessor
preprocessor = ProcessTracksForTrackAutoencoder(
    num_support_tracks=2048,
    num_target_tracks=2048,
    video_key="video",
    before_boundary=True,
)

# Preprocess Batch
batch = {
    "video": video,
    "tracks": einops.rearrange(
        transforms.convert_grid_coordinates(
            tracks + 0.5, (width, height), (1, 1)
        ),
        "q t c -> t q c",
    ),
    "visible": einops.rearrange(visibles, "q t -> t q"),
}

batch = preprocessor.random_map(batch)
batch.pop("tracks", None)

# Run forward pass
outputs = forward(params, batch)

height, width = video.shape[1:3]

reconstructed_tracks = transforms.convert_grid_coordinates(
    outputs.tracks[0], (1, 1), (width, height)
)

support_tracks_vis = transforms.convert_grid_coordinates(
    batch['support_tracks'][0], (1, 1), (width, height)
)

target_tracks_vis = transforms.convert_grid_coordinates(
    batch['target_points'][0], (1, 1), (width, height)
)

reconstructed_visibles = model_utils.postprocess_occlusions(
    outputs.visible_logits, outputs.certain_logits
)

# NOTE: uncomment the lines below to also visualize the support & target tracks.
video_length = video.shape[0]

print(len(support_tracks_vis))
print(len(target_tracks_vis))

print(len(support_tracks_vis[:, :video_length]), len(target_tracks_vis[:, :video_length]))
print(len(batch['support_tracks_visible'][0, :, :video_length]), len(batch['target_tracks_visible'][0, :, :video_length]))

video_viz = viz_utils.paint_point_track(
    video,
    support_tracks_vis[:, :video_length],
    batch['support_tracks_visible'][0, :, :video_length],
)
media.write_video(
    "support_tracks.mp4",
    video_viz,
    fps=FPS,
)

# media.show_video(video_viz, fps=FPS)

video_viz = viz_utils.paint_point_track(
    video,
    target_tracks_vis[:, :video_length],
    batch['target_tracks_visible'][0, :, :video_length],
)
media.write_video(
    "target_tracks.mp4",
    video_viz,
    fps=FPS,
)

video_viz = viz_utils.paint_point_track(
    video,
    reconstructed_tracks[:, :video_length],
    batch['target_tracks_visible'][0, :, :video_length],
)

# save video
media.write_video(
    "reconstructed_tracks.mp4",
    video_viz,
    fps=FPS,
)

# Query from the first frame onward.
query_points = np.zeros((
    reconstructed_visibles.shape[0],
    batch['target_tracks_visible'].shape[1],
    1,
))

# Compute TapVid metrics
metrics = compute_tapvid_metrics(
    query_points=query_points,
    gt_occluded=1 - batch['target_tracks_visible'][..., :video_length, 0],
    gt_tracks=target_tracks_vis[None, ..., :video_length, :],
    pred_occluded=reconstructed_visibles[..., :video_length, 0],
    pred_tracks=reconstructed_tracks[..., :video_length, :],
    query_mode='strided',
    get_trackwise_metrics=False,
)

jaccard = np.mean([metrics[f'jaccard_{d}'] for d in [1, 2, 4, 8, 16]])
print('Average Jaccard:', jaccard)
print('Occlusion Accuracy:', metrics['occlusion_accuracy'].mean())