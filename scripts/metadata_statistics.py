#!/usr/bin/env python3
from collections import Counter
import csv
import itertools
import os
import yaml

MAIN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METADATA_FILE = os.path.join(MAIN_FOLDER, 'data/metadata.csv')
ANNOTATION_INPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotations.yaml')
STATISTICS_OUTPUT_FILE = os.path.join(MAIN_FOLDER, 'data/annotation_statistics.yaml')

if __name__ == '__main__':
    metadata = {}
    with open(METADATA_FILE) as f:
        csv_reader = csv.reader(f, delimiter=',')
        keys = next(csv_reader)
        for s in csv_reader:
            data = dict(zip(keys, s))
            set_id = int(data.pop('Set-ID'))
            metadata[set_id] = data

    # remove set id from keys
    keys.pop(0)

    with open(ANNOTATION_INPUT_FILE) as f:
        annotations = yaml.safe_load(f)

    image_names = annotations['labels'].keys()
    set_counts = Counter(list(map(lambda v: int(v.split('-', 1)[0]), image_names)))

    statistics = {}
    for key in keys:
        current_stats = {}
        for set_id, set_data in metadata.items():
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
