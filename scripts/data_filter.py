#!/usr/bin/env python3
import os
import yaml

INPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_raw'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
INPUT_FILE = os.path.join(INPUT_FOLDER, 'annotations.yaml')
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'annotations.yaml')

if __name__ == '__main__':
    print('Reading all annotations')
    with open(INPUT_FILE) as f:
        all_annotations = yaml.safe_load(f)

    print('Filtering the annotations')
    filtered_annotations = {
        'sets': all_annotations['sets'],
        'images': {},
    }
    for image in os.listdir(OUTPUT_FOLDER):
        if image in all_annotations['images']:
            filtered_annotations['images'][image] = all_annotations['images'][image]

    print('Writing the annotations')
    with open(OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(filtered_annotations))

    print(f'The remaining annotations have been written to {OUTPUT_FILE}')
