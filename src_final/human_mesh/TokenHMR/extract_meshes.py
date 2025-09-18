import os
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_meshes_from_tracking(tracking_pkl_path, output_dir):
    """
    Extract all meshes from tracking results and save them as separate files.
    
    Args:
        tracking_pkl_path: Path to the tracking results pickle file
        output_dir: Directory to save the extracted meshes
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tracking results
    tracking_data = joblib.load(tracking_pkl_path)
    
    # Get all frames
    frames = sorted(tracking_data.keys())
    
    # Process each frame
    for frame_idx, frame_name in enumerate(tqdm(frames, desc="Extracting meshes")):
        frame_data = tracking_data[frame_name]
        
        # Skip if no tracks in this frame
        if len(frame_data['tid']) == 0:
            continue
            
        # Get number of tracks in this frame
        num_tracks = len(frame_data['tid'])
        
        # Process each track
        for track_idx in range(num_tracks):
            track_id = frame_data['tid'][track_idx]
            
            # Get mesh data
            smpl_params = frame_data['smpl'][track_idx]
            camera = frame_data['camera'][track_idx]
            camera_bbox = frame_data['camera_bbox'][track_idx]
            
            # Create output filename
            output_filename = f"frame_{frame_idx:04d}_track_{track_id:04d}.npz"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save mesh data
            np.savez(
                output_path,
                smpl_params=smpl_params,
                camera=camera,
                camera_bbox=camera_bbox,
                frame_idx=frame_idx,
                track_id=track_id,
                frame_name=frame_name
            )

if __name__ == "__main__":
    # Example usage
    tracking_pkl_path = "path/to/tracking_results.pkl"
    output_dir = "path/to/output/meshes"
    extract_meshes_from_tracking(tracking_pkl_path, output_dir) 