from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
import os

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": True, "full_frame": True}
)
print(mesh_generator)

DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes"
OUTPUT_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes_mesh"
actions = os.listdir(DIR)
actions = ["BodyWeightSquats", "HulaHoop", "JumpingJack", "PullUps", "PushUps", "Shotput", "SoccerJuggling", "TennisSwing", "ThrowDiscus", "WallPushups"]
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

# DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5"
# OUTPUT_DIR = "videos/ucf101/mesh_runway_gen4_videos"
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

# DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen3_alpha_videos_5"
# OUTPUT_DIR = "videos/ucf101/mesh_runway_gen3_alpha_videos"
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

# DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/cogvideox_videos_5"
# OUTPUT_DIR = "videos/ucf101/mesh_cogvideox_videos"
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


# DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5"
# OUTPUT_DIR = "videos/ucf101/mesh_wan21_videos"
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

# DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/hunyuan_videos_360p_formatted"
# OUTPUT_DIR = "videos/ucf101/mesh_hunyuan_360p_videos"
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

# DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/opensora_videos_256p_formatted"
# OUTPUT_DIR = "videos/ucf101/mesh_opensora_256p_videos"
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