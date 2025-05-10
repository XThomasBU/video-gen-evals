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
