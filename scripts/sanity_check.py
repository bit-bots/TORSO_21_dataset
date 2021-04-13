#!/usr/bin/env python3
from collections import Counter, defaultdict
import os
import yaml

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations.yaml')

if __name__ == '__main__':
    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)['images']

    error_count = 0

    for image_name, image_data in annotations.items():
        image_annotations = image_data['annotations']
        field_edge = False
        in_image = {}
        for annotation in image_annotations:
            at = annotation['type']
            if annotation['in_image']:
                if in_image.get(at, True) == False:
                    # it was not in image but now in image
                    print(f"{image_name}: {annotation['type']} not in image and in image")
                    error_count += 1
                else:
                    in_image[at] = True
            else:
                if in_image.get(at, False) == True:
                    # it was in image and is now not in image
                    print(f"{image_name}: {annotation['type']} not in image and in image")
                    error_count += 1
                else:
                    in_image[at] = False

            if at == 'field edge':
                field_edge = True

        if not field_edge:
            print(f"No field edge in image {image_name}")
            error_count += 1

    if error_count > 0:
        print(f"Found {error_count} errors in {len(annotations)} images!")
    else:
        print(f"No errors found.")

