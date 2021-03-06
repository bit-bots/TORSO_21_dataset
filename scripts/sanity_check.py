#!/usr/bin/env python3
from collections import Counter, defaultdict
import os
import sys
import yaml

ALL_TYPES = set(('T-Intersection', 'X-Intersection', 'L-Intersection', 'robot', 'ball', 'goalpost', 'field edge'))

if len(sys.argv) != 2:
    sys.exit(f'Usage: {sys.argv[0]} annotations.yaml')

ANNOTATION_INPUT_FILE = sys.argv[1]

if __name__ == '__main__':
    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)['images']

    not_in_image = 0
    invalid_vector = 0
    field_edge_not_in_image = 0
    missing_types = 0

    for image_name, image_data in annotations.items():
        image_annotations = image_data['annotations']
        image_id = image_data['id']
        field_edge = False
        in_image = {}
        for annotation in image_annotations:
            at = annotation['type']
            if annotation['in_image']:
                if in_image.get(at, True) == False:
                    # it was not in image but now in image
                    print(f"{image_id} ({image_id} ({image_name})): {annotation['type']} not in image and in image")
                    not_in_image += 1
                else:
                    in_image[at] = True
            else:
                if in_image.get(at, False) == True:
                    # it was in image and is now not in image
                    print(f"{image_id} ({image_name}): {annotation['type']} not in image and in image")
                    not_in_image += 1
                else:
                    in_image[at] = False

            if annotation['in_image']:
                # Check vector out of bounds
                if not all(0 <= x <= image_data['width'] and
                           0 <= y <= image_data['height']
                           for x, y in annotation['vector']):
                    print(f"{image_id} ({image_name}) has an invalid {annotation['type']} annotation (vector: {annotation['vector']}, size: {image_data['width']}x{image_data['height']})")
                    invalid_vector += 1

            if at == 'field edge' and not annotation['in_image']:
                print(f"{image_id} ({image_name}): field edge should be in image!")
                field_edge_not_in_image += 1

        types_in_image = set(map(lambda a: a['type'], image_annotations))
        if not types_in_image.issuperset(ALL_TYPES):
            print(f"{image_id} ({image_name}) is missing annotations for {' and '.join(ALL_TYPES - types_in_image)}")
            missing_types += 1

    error_count = sum((not_in_image, invalid_vector, field_edge_not_in_image, missing_types))
    if error_count > 0:
        print(f"Found {error_count} errors in {len(annotations)} images!")
        print(f"{not_in_image} conflicting not in image / in image annotations")
        print(f"{invalid_vector} vectors with points outside of the image")
        print(f"{field_edge_not_in_image} field edges labeled as not in image")
        print(f"{missing_types} images with missing annotation types")
    else:
        print(f"No errors found.")

