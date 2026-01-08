import os

import yaml

source_dir = "/srv/ssd_nvm/dataset/TORSO21/reality"
target_dir = "data/reality_rfdetr"

splits = [
    "train",
    "test",
    "valid",
]

for split in splits:
    print(split)
    with open(os.path.join(target_dir, split, "annotations.yaml")) as f:
        annotations = yaml.safe_load(f)

    for image_name in annotations["images"].keys():
        for data_type in ["images"]:  # , "segmentations"]:
            # Take valid images from train folder
            source_split = split
            if split == "valid":
                source_split = "train"

            source_path = os.path.join(source_dir, source_split, data_type, image_name)
            target_path = os.path.join(target_dir, split, image_name)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if not os.path.exists(target_path):
                # print(f"Creating symlink: {target_path} -> {source_path}")
                os.symlink(source_path, target_path)
