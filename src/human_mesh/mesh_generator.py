from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
import os

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": True, "full_frame": True}
)
print(mesh_generator)

DIR = "saved_data/ucf101"
videos = os.listdir(DIR)

for video in videos:
    mesh_generator.generate_mesh_from_frames(
        os.path.join(DIR, video, "full_frames")
    )

for video in videos:
    mesh_generator.generate_mesh_from_frames(
        os.path.join(DIR, video, "generated_videos_cogvideox", "frames")
    )


for video in videos:
    mesh_generator.generate_mesh_from_frames(
        os.path.join(DIR, video, "generated_videos_runway_gen4_turbo", "frames")
    )



# mesh_generator.generate_mesh_from_frames(
#     "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/generated_videos_cogvideox/frames"
# )