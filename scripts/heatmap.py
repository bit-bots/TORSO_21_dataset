#!/usr/bin/env python3
import os
import numpy as np
import cv2
import time
import pickle
import functools
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm


def render(root, annotations, annotation_type, size, canvas_template, f):
    canvas = np.zeros_like(canvas_template)

    img_path = os.path.join(root, f)

    img = cv2.imread(img_path)

    shape = img.shape

    annotations_of_type_in_image = filter(
        lambda annotation: 
            annotation_type in annotation['type'].lower() and annotation['in_image'],
        annotations[f])

    for annotation in annotations_of_type_in_image:

        vector = np.array(annotation['vector'], dtype = np.float64)

        vector[:, 0] = vector[:, 0] / float(shape[1])
        vector[:, 1] = vector[:, 1] / float(shape[0])

        vector = np.multiply(vector, size).astype(np.int32)

        if annotation_type == 'ball':
            canvas += cv2.circle(
                np.zeros_like(canvas), 
                (
                    (vector[0][0] + vector[1][0]) // 2, 
                    (vector[0][1] + vector[1][1]) // 2
                ), 
                ((vector[1][0] - vector[0][0]) + (vector[1][1] - vector[0][1])) // 4,
                1, -1)
        if annotation_type == 'robot':
            canvas += cv2.rectangle(np.zeros_like(canvas), tuple(vector[1].tolist()), tuple(vector[0].tolist()), 1, -1)
        elif annotation_type == 'goalpost':
            canvas += cv2.fillConvexPoly(np.zeros_like(canvas), vector, 1.0)
        elif annotation_type == 'field edge':
            canvas += cv2.fillConvexPoly(np.zeros_like(canvas), np.array([[0,size]] + vector.tolist() + [[size, size]]), 1.0)
        elif 'intersection' in annotation_type:
            canvas += cv2.circle(
                np.zeros_like(canvas), 
                (vector[0][0], vector[0][1]), 
                5, 1, -1)
    return canvas



class LineLabelTool(object):
    def __init__(self):
        self.path="/home/florian/Downloads/imageset/"
        self.annotation_path="/home/florian/Downloads/vision_dataset_2021_labels.pickle"
        self.pool = multiprocessing.Pool()

    def calc_heatmap(self, annotation_type, size):

        annotation_type = annotation_type.lower()

        canvas = np.zeros((size,size), dtype=np.float64)

        with open(self.annotation_path, 'rb') as f:
            annotations = pickle.load(f)

        for root,_,f_names in os.walk(self.path):
            
            f_names = sorted([f for f in f_names if f.endswith(".png") or f.endswith(".jpg")])

            out_list = self.pool.map(
                functools.partial(
                    render, 
                    root, 
                    annotations['labels'], 
                    annotation_type, 
                    size, 
                    canvas), 
                f_names)

            canvas += np.array(out_list).sum(axis=0)

        return canvas / len(f_names)

    def main(self):
        classes = ['Ball', 'Goalpost', 'Robot', 'Field Edge', 'T-Intersection', 'L-Intersection', 'X-Intersection']
        size = 100

        plt.figure(figsize=(8,3))
        sns.set_style("whitegrid")
        sns.set_context("paper")
        plt.rcParams["font.sans-serif"] = "arial"
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        for idx, cls in enumerate(classes):
            sub = plt.subplot(2, 4, idx + 1, xticks=[], yticks=[])
            sub.set_title(f'{cls}')
            heatmap = self.calc_heatmap(cls, size)
            sns.heatmap(heatmap, xticklabels=False, yticklabels=False, linewidths=0.0, rasterized=True)

        plt.tight_layout()
        plt.savefig("heatmaps.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.show()


if __name__ == "__main__":
    LineLabelTool().main()
