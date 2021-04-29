#!/usr/bin/env python3
from collections import Counter, defaultdict
import os
import sys
import yaml

if len(sys.argv) != 2:
    sys.exit(f'Usage: {sys.argv[0]} annotations.yaml')

ANNOTATION_INPUT_FILE = sys.argv[1]
STATISTICS_OUTPUT_FILE = os.path.join(os.path.dirname(sys.argv[1]), 'annotation_statistics.yaml')

if __name__ == '__main__':
    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)['images']

    # type count is a dict of dicts
    # the outer dict maps type to dict, the inner dict maps count to number of images with this count
    type_count = defaultdict(lambda: defaultdict(lambda: 0))

    for image_name, image_data in annotations.items():
        image_annotations = image_data['annotations']
        type_count_image = defaultdict(lambda: 0)
        for annotation in image_annotations:
            if annotation['in_image']:
                type_count_image[annotation['type']] += 1
            else:
                type_count_image[annotation['type']] = 0

        for type_name, number_in_image in type_count_image.items():
            type_count[type_name][number_in_image] += 1

    # convert to normal dict
    d = {k: dict(v) for k, v in type_count.items()}
    with open(STATISTICS_OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(d))
