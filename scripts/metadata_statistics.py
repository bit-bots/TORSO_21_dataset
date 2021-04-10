#!/usr/bin/env python3
from collections import Counter
import itertools
import os
import yaml

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data_raw/annotations_with_metadata.yaml')
STATISTICS_OUTPUT_FILE = os.path.join(MAIN_FOLDER, 'data_raw/annotation_statistics.yaml')

if __name__ == '__main__':
    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)

    metadata_keys = set(itertools.chain(*(d.keys() for d in annotations['sets'].values())))
    metadata_keys.discard('description')
    metadata_keys.discard('name')
    metadata_keys.discard('location')  # location is already in the csv file
    image_names = annotations['images'].keys()
    set_counts = Counter(list(map(lambda v: int(v.split('-', 1)[0]), image_names)))

    statistics = {}
    for key in metadata_keys:
        current_stats = {}
        for set_id, set_data in annotations['sets'].items():
            if key in set_data:
                val = set_data[key]
                if val in current_stats:
                    current_stats[val] += set_counts[set_id]
                else:
                    current_stats[val] = set_counts[set_id]
        statistics[key] = current_stats

    # Filter out zeros
    for stat, data in statistics.items():
        keys_to_delete = []
        for k, v in data.items():
            if v == 0:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del data[k]

    with open(STATISTICS_OUTPUT_FILE, 'w') as f:
        f.write(yaml.dump(statistics))
