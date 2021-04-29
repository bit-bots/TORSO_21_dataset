#!/usr/bin/env python3
from collections import defaultdict
import os
import sys
import yaml

if len(sys.argv) != 2:
    sys.exit(f'Usage: {sys.argv[0]} annotations.yaml')

ANNOTATION_INPUT_FILE = sys.argv[1]
STATISTICS_OUTPUT_FILE = os.path.join(os.path.dirname(sys.argv[1]), 'metadata_statistics.yaml')

if __name__ == '__main__':
    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)

    metadata_count = defaultdict(lambda: defaultdict(lambda: 0))
    image_names = annotations['images'].keys()

    for image in annotations['images'].values():
        for key, value in image['metadata'].items():
            metadata_count[key][value] += 1


    d = {k: dict(v) for k, v in metadata_count.items()}
    with open(STATISTICS_OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(d))
