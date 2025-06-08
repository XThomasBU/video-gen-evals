from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
import os

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": True, "full_frame": True}
)
print(mesh_generator)

DIR = "videos/ucf101/real_videos_5"
GEN_DIR = "videos/ucf101/runway_gen4_videos_5"
OUTPUT_DIR = "videos/ucf101/mesh_real_videos_5"
actions = os.listdir(GEN_DIR)

for action in actions:
    videos = os.listdir(os.path.join(GEN_DIR, action))
    for video in videos:
        input_folder_path = os.path.join(DIR, action, video)
        out_dir = os.path.join(OUTPUT_DIR, action, video, "tokenhmr_mesh")
        mesh_generator.generate_mesh_from_frames(
            input_folder_path, out_dir
        )

DIR = "videos/ucf101/runway_gen4_videos_5"
OUTPUT_DIR = "videos/ucf101/mesh_runway_gen4_videos_5"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        input_folder_path = os.path.join(DIR, action, video)
        out_dir = os.path.join(OUTPUT_DIR, action, video, "tokenhmr_mesh")
        # skip if out_dir/mesh_overlay.mp4 exists
        if os.path.exists(os.path.join(out_dir, "mesh_overlay.mp4")):
            print(f"Skipping {input_folder_path} because mesh_overlay.mp4 already exists")
            continue
        print(f"Generating mesh for {input_folder_path} to {out_dir}")
        mesh_generator.generate_mesh_from_frames(
            input_folder_path, out_dir
        )

# DIR = "videos/ucf101/cogvideox_videos_5"
# OUTPUT_DIR = "videos/ucf101/mesh_cogvideox_videos_5"
# videos = os.listdir(DIR)
# actions = os.listdir(DIR)

# for action in actions:
#     videos = os.listdir(os.path.join(DIR, action))
#     for video in videos:
#         input_folder_path = os.path.join(DIR, action, video)
#         out_dir = os.path.join(OUTPUT_DIR, action, video, "tokenhmr_mesh")
#         # skip if out_dir/mesh_overlay.mp4 exists
#         if os.path.exists(os.path.join(out_dir, "mesh_overlay.mp4")):
#             print(f"Skipping {input_folder_path} because mesh_overlay.mp4 already exists")
#             continue
#         print(f"Generating mesh for {input_folder_path} to {out_dir}")
#         mesh_generator.generate_mesh_from_frames(
#             input_folder_path, out_dir
#         )



# mesh_generator.generate_mesh_from_frames(
#     "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/generated_videos_cogvideox/frames"
# )