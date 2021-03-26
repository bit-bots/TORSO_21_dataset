#!/usr/bin/env python3
import json
import os
import yaml
from zipfile import ZipFile

INPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_raw'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
ANNOTATION_INPUT_FILE = os.path.join(INPUT_FOLDER, 'annotations.yaml')
SELECTION_INPUT_FILE = os.path.join(INPUT_FOLDER, 'selection.yaml')
IMAGES_ZIP_FILE = os.path.join(OUTPUT_FOLDER, 'images.zip')
ANNOTATION_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'annotations.txt')

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print('Writing images to zip file')
    with open(SELECTION_INPUT_FILE) as f:
        selection = yaml.safe_load(f)['selection']

    all_images = os.listdir(INPUT_FOLDER)
    with ZipFile(IMAGES_ZIP_FILE, 'w') as f:
        for image in selection:
            if image not in all_images:
                print(f'!!! Image {image} in selection but not in data folder !!!')
            else:
                f.write(os.path.join(INPUT_FOLDER, image), image)

    print('Reading all existing labels')
    with open(ANNOTATION_INPUT_FILE) as f:
        data = yaml.safe_load(f)

    print('Writing labels in upload format')
    with open(ANNOTATION_OUTPUT_FILE, 'w') as f:
        for image, annotations in data['images'].items():
            if image not in selection:
                continue
            for annotation in annotations:
                if annotation['in_image']:
                    vector = {}
                    for i, (x, y) in enumerate(annotation['vector']):
                        vector['x' + str(i + 1)] = x
                        vector['y' + str(i + 1)] = y

                    f.write(image + '|' + 
                            annotation['type'] + '|' +
                            json.dumps(vector) + '|' +
                            ('b' if annotation['blurred'] else '') +
                            ('c' if annotation['concealed'] else '') +
                            '\n')
                else:
                    f.write(image + '|' +
                            annotation['type'] + '|' +
                            'not in image' + 
                            '\n')

    print(f'The images have been written to {IMAGES_ZIP_FILE}')
    print(f'The annotations have been written to {ANNOTATION_OUTPUT_FILE}')
