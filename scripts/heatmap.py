#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import yaml
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


class LineLabelTool(object):
    def __init__(self):
        self.path="/home/florian/Downloads/imageset/"
        self.annotation_path="/home/florian/Downloads/vision_dataset_2021_labels.pickle"

    def main_loop(self):

        annotation_type = "robot"

        size = 50

        canvas = np.zeros((size,size), dtype=np.float64)

        annotation_count = 0

        img_count = 0

        with open(self.annotation_path, 'rb') as f:
            annotations = pickle.load(f)

        for root,_,f_names in os.walk(self.path):
            
            f_names = sorted([f for f in f_names if f.endswith(".png") or f.endswith(".jpg")])

            for f in tqdm(f_names):
                img_path = os.path.join(root, f)

                img = cv2.imread(img_path)

                shape = img.shape
                
                img_count += 1

                annotations_of_type_in_image = filter(
                    lambda annotation: annotation_type in annotation['type'] and annotation['in_image'],
                     annotations['labels'][f])

                for annotation in annotations_of_type_in_image:
                    annotation_count += 1

                    vector = np.array(annotation['vector'], dtype = np.float64)

                    vector[:, 0] = vector[:, 0] / float(shape[1])
                    vector[:, 1] = vector[:, 1] / float(shape[0])

                    vector = np.multiply(vector, size).astype(np.int32)

                    if annotation_type in ['ball', 'robot']:
                        canvas += cv2.rectangle(np.zeros_like(canvas), tuple(vector[1].tolist()), tuple(vector[0].tolist()), 1, -1)
                    elif annotation_type in ['goalpost']:
                        canvas += cv2.fillConvexPoly(np.zeros_like(canvas), vector, 1.0)
                    elif annotation_type in ['field edge']:
                        canvas += cv2.fillConvexPoly(np.zeros_like(canvas), np.array([[0,size]] + vector.tolist() + [[size, size]]), 1.0)
                    elif 'Intersection' in annotation_type:
                        canvas[vector[0][1], vector[0][0]] += 1

                #if img_count > 2000:
                #    break

        sns.heatmap(canvas / img_count, xticklabels=False, yticklabels=False)
        plt.show()
        print(canvas.astype(np.int32))


if __name__ == "__main__":
    LineLabelTool().main_loop()
