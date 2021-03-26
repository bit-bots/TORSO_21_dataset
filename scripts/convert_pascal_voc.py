#!/usr/bin/env python3
""" This script converts labels from the Pascal VOC XML format to the `yaml` format as defined in the Readme."""

import os
import yaml
import xml.etree.ElementTree as ET


BASE_DIR = '/tmp/voc/voc_labels'
ANNOTATION_OUTPUT_FILE = os.path.join(BASE_DIR, 'annotations.yaml')

NAME_TO_TYPE = {
    'ball': "ball",
    'goal post': "goalpost",
    'rhoban': "robot",
}

BLURRED_DEFAULT = False
CONCEALED_DEFAULT = False


def get_data_from_VOC(path):
    tree = ET.parse(path)
    root = tree.getroot()  # Root of the VOC format

    image = root.find('filename').text

    image_data = {
        'annotations': get_annotations(root),
        'metadata': get_metadata(root),
    }
    return image, image_data

def get_annotations(subtree):
    annotations = []

    # Find all annotations (called objects)
    for annotation in subtree.findall('object'):
        bbox = annotation.find('bndbox')
        annotation_data = {
            'type': NAME_TO_TYPE[annotation.find('name').text],
            'in_image': True,
            'blurred': BLURRED_DEFAULT,
            'concealed': CONCEALED_DEFAULT,
            'vector': [  # BBox points
                [  # X, Y value
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                ],
                [  # X, Y value
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                ]
            ],
        }
        annotations.append(annotation_data)

    # Check for missing types
    # Add annotations for types not in image
    ball = goalpost = robot = False
    for annotation in annotations:
        if annotation['type'] == 'ball':
            ball = True
        elif annotation['type'] == 'goalpost':
            goalpost = True
        elif annotation['type'] == 'robot':
            robot = True

    if not ball:
        annotations.append({'type': 'ball', 'in_image': False})
    if not goalpost:
        annotations.append({'type': 'goalpost', 'in_image': False})
    if not robot:
        annotations.append({'type': 'robot', 'in_image': False})

    return annotations

def get_metadata(subtree):

    size = subtree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    return {
        'width': width,
        'height': height,
        # 'depth': depth,  # TODO: Do we need this?
    }


if __name__ == '__main__':
    data = {}  # This collects all data, that will be saved as yaml
    data['images'] = {}

    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".xml"):  # VOC file format
            image, image_data = get_data_from_VOC(os.path.join(BASE_DIR, filename))
            data['images'][image] = image_data

    with open(ANNOTATION_OUTPUT_FILE, 'w') as f:
        yaml.dump(data, f)
