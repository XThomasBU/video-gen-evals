from .TokenHMR.mesh_generator import TokenHMRMeshGenerator

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": True, "full_frame": True}
)
print(mesh_generator)

mesh_generator.generate_mesh_from_frames(
    "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/hacs/x_gEdkM6kwE/full_frames"
)
