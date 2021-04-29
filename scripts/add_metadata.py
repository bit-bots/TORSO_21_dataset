#!/usr/bin/env python3
import csv
import os
import yaml

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations.yaml')
ANNOTATION_TRAIN_OUTPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations_train.yaml')
ANNOTATION_TEST_OUTPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations_test.yaml')
METADATA_FILE = os.path.join(MAIN_FOLDER, 'data/metadata.csv')
TRAIN_LIST = os.path.join(MAIN_FOLDER, 'data/train_images.txt')
TEST_LIST = os.path.join(MAIN_FOLDER, 'data/test_images.txt')

if __name__ == '__main__':
    print(f'Adding metadata from {METADATA_FILE} to {ANNOTATION_INPUT_FILE}')

    metadata = {}
    with open(METADATA_FILE) as f:
        csv_reader = csv.reader(f, delimiter=',')
        keys = next(csv_reader)
        for s in csv_reader:
            data = dict(zip(keys, s))
            set_id = int(data.pop('Set-ID'))
            metadata[set_id] = data

    with open(TRAIN_LIST) as f:
        train_images = f.read().splitlines()

    with open(TEST_LIST) as f:
        test_images = f.read().splitlines()


    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)

    new_annotations_train = {}
    new_annotations_test = {}

    for image_name in annotations['images']:
        current_set = int(image_name.split('-')[0])
        image_data = annotations['images'][image_name]
        image_data['metadata'] = metadata[current_set]
        if image_name in train_images:
            new_annotations_train[image_name] = image_data
        elif image_name in test_images:
            new_annotations_test[image_name] = image_data

    output_train = {}
    output_train["images"] = new_annotations_train
    with open(ANNOTATION_TRAIN_OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(output_train))

    output_test = {}
    output_test["images"] = new_annotations_test
    with open(ANNOTATION_TEST_OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(output_test))
