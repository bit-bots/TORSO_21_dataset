#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import tqdm
import yaml


def extract_classes(data: dict) -> list[dict[str, int | str]]:
    types: set[str] = set()

    for image in data["images"].values():
        for annotation in image.get("annotations", []):
            types.add(annotation["type"])

    unwanted_types = {"X-Intersection", "T-Intersection", "L-Intersection"}
    types = types - unwanted_types

    print(
        f"Found {len(types)} unique classes. Removed unwanted types: {unwanted_types}. Resulting classes: {types}"
    )

    categories = [
        {"id": idx, "name": class_type, "supercategory": "none"}
        for idx, class_type in enumerate(sorted(types))
    ]

    return categories


def convert(data: dict, categories: list) -> tuple[list, list]:
    images = []
    annotations = []
    annotation_id = 1
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

    for image_name, image in tqdm.tqdm(data["images"].items()):
        images.append(
            {
                "id": image["id"],
                "file_name": image_name,
                "width": image["width"],
                "height": image["height"],
            }
        )

        for annotation in image.get("annotations", []):
            if annotation["type"] not in category_name_to_id:
                continue

            if not annotation["in_image"]:
                continue

            # Goalpost is a polygon, others are bounding boxes
            # TORSO-21 dataset uses list of points in x, y image coordinates
            # vector: [[x1, y1], [x2, y2]] for bounding boxes
            # vector: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] for goalposts
            # COCO uses absolute bounding boxes [x, y, width, height] (top-left corner)
            points = annotation["vector"]

            if annotation["type"] == "goalpost":
                # Polygon segmentation for goalposts
                # Flatten the list of points
                segmentation = []
                for point in points:
                    segmentation.append(point[0])
                    segmentation.append(point[1])

                # Bounding box for goalposts
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min = min(x_coords)
                y_min = min(y_coords)
                width = max(x_coords) - x_min
                height = max(y_coords) - y_min

            else:
                # Bounding box for other types
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                width = abs(points[1][0] - points[0][0])
                height = abs(points[1][1] - points[0][1])

                # Segmentation as a rectangle
                segmentation = [
                    x_min,
                    y_min,
                    x_min + width,
                    y_min,
                    x_min + width,
                    y_min + height,
                    x_min,
                    y_min + height,
                ]

            # COCO bbox format
            bbox = [x_min, y_min, width, height]

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image["id"],
                    "category_id": category_name_to_id[annotation["type"]],
                    "bbox": bbox,
                    "segmentation": [segmentation],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return images, annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TORSO-21 dataset from YAML to COCO format JSON."
    )
    parser.add_argument("yaml_path", type=Path, help="Path to the input YAML file.")
    parser.add_argument(
        "output_json_path", type=Path, help="Path to the output COCO JSON file."
    )
    args = parser.parse_args()

    with open(args.yaml_path, "r") as f:
        data = yaml.safe_load(f)

    categories = extract_classes(data)
    images, annotations = convert(data, categories)

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {
            "description": "TORSO-21 Dataset in COCO format",
            "version": "1.0",
        },
        "licenses": [],
    }

    with open(args.output_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)
