#!/usr/bin/env python3
import csv
import os
import yaml

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations.yaml')
ANNOTATION_OUTPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations_with_metadata.yaml')
METADATA_FILE = os.path.join(MAIN_FOLDER, 'data/metadata.csv')

if __name__ == '__main__':
    print(f'Adding metadata from {METADATA_FILE} to {ANNOTATION_INPUT_FILE}, writing to {ANNOTATION_OUTPUT_FILE}')

    metadata = {}
    with open(METADATA_FILE) as f:
        csv_reader = csv.reader(f, delimiter=',')
        keys = next(csv_reader)
        for s in csv_reader:
            data = dict(zip(keys, s))
            set_id = int(data.pop('Set-ID'))
            metadata[set_id] = data

    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)

    for image_name, image_data in annotations['images']:
        current_set = int(image_name.split('-')[0])
        image_data['metadata'] = metadata[current_set]

    with open(ANNOTATION_OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(annotations))

