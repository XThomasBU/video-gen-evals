from .TokenHMR.mesh_generator import TokenHMRMeshGenerator
import os
import json

mesh_generator = TokenHMRMeshGenerator(
    config={"side_view": True, "save_mesh": True, "full_frame": True}
)
print(mesh_generator)


DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes"
videos = os.listdir(DIR)
actions = os.listdir(DIR)

single_person_dict = {}
for action in actions:
    videos = os.listdir(os.path.join(DIR, action))
    single_person_dict[action] = []
    for video in videos:
        input_folder_path = os.path.join(DIR, action, video)
        single_person = mesh_generator.filter_single_person(input_folder_path)
        if single_person:
            single_person_dict[action].append(video)
        print(f"{input_folder_path}: {single_person}")


with open("ucf101_single_person_videos.json", "w") as f:
    json.dump(single_person_dict, f, indent=4)
            