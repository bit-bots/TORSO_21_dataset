#!/usr/bin/env python3
import json
import os
import yaml
from zipfile import ZipFile

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
ANNOTATION_INPUT_FILE = os.path.join(DATA_FOLDER, 'annotations.yaml')
IMAGES_ZIP_FILE = os.path.join(DATA_FOLDER, 'images.zip')
ANNOTATION_OUTPUT_FILE = os.path.join(DATA_FOLDER, 'annotations.txt')

if __name__ == '__main__':
    print('Writing images to zip file')
    with ZipFile(IMAGES_ZIP_FILE, 'w') as f:
        for image in os.listdir(DATA_FOLDER):
            if image.endswith('.zip') or image.endswith('.yaml'):
                continue
            f.write(os.path.join(DATA_FOLDER, image), image)

    print('Writing labels in upload format')
    with open(ANNOTATION_INPUT_FILE) as f:
        data = yaml.safe_load(f)

    with open(ANNOTATION_OUTPUT_FILE, 'w') as f:
        for image, image_data in data['images'].items():
            for annotation in image_data['annotations']:
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
