#!/usr/bin/env python3
import getpass
import os
import sys
import yaml
from download import login, download_zip, download_annotations

SETS = [261, 156, 614, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035]
EXPORT_FORMAT = 196
OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_raw'))
SET_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '%s'))

if __name__ == '__main__':
    print(f'The merged files will be saved to {OUTPUT_FOLDER}')

    username = input('ImageTagger username: ')
    password = getpass.getpass()
    login_data = login(username, password)
    if not login_data:
        sys.exit('Login failed')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    annotation_file = os.path.join(OUTPUT_FOLDER, 'annotations.yaml')

    for s in SETS:
        set_folder = SET_FOLDER % s
        print(f'Downloading imageset {s} to {set_folder}')
        download_zip(s, set_folder, login_data)

        print(f'Moving images of set {s}')
        for image in os.listdir(str(s)):
            os.rename(os.path.join(str(s), image), os.path.join(OUTPUT_FOLDER, f'{s}-{image}'))
        
        print(f'Downloading annotations for set {s}')
        download_annotations(s, EXPORT_FORMAT, set_folder, login_data)

        print(f'Converting annotations for set {s}')
        # Read annotations for this set
        with open(f'{s}/{s}.txt') as f:
            new_annotation_data = yaml.safe_load(f)

        # Load existing collected annotations
        if os.path.isfile(annotation_file):
            with open(annotation_file) as f:
                annotation_data = yaml.safe_load(f)
        else:
            annotation_data = {'sets': {}, 'images': {}}

        # Add metadata for this set
        annotation_data['sets'][s] = new_annotation_data['metadata']

        # Add the images of this set, prepend them with the set id
        if new_annotation_data['labels']:
            for image, annotations in new_annotation_data['labels'].items():
                new_name = f"{s}-{image}"
                annotation_data['images'][new_name] = annotations

        # Write the new data
        with open(annotation_file, 'w') as f:
            f.write(yaml.dump(annotation_data))

        print(f'Finished processing set {s}')

    print(f'All data has been moved to {OUTPUT_FOLDER}')
    print(f'The annotations are located in {annotation_file}')
