from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
import os

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": True, "full_frame": True}
)
print(mesh_generator)


DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5"
OUTPUT_DIR = "videos/ucf101/mesh_runway_gen4_videos"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    for video in videos:
        input_folder_path = os.path.join(DIR, action, video)
        single_person = mesh_generator.filter_single_person(input_folder_path)
        print(single_person)
        exit()