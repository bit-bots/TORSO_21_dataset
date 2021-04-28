#!/usr/bin/env python3
import os
import numpy as np
import cv2
import time
import pickle
import yaml
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
        annotations[f]["annotations"])

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

def render_lines(lines_root, canvas_template, f):
    line_path = os.path.join(lines_root, f)
    line = cv2.imread(line_path)
    if line is not None:
        line = cv2.resize(line, dsize=canvas_template.shape, interpolation=cv2.INTER_NEAREST)
        return line[:,:,0]
    else:
        print(f"Warning: Line annotation does not exist: '{line_path}'")
        return np.zeros_like(canvas_template)



class Heatmapper(object):
    def __init__(self, image_path, annotation_path, lines_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.lines_path = lines_path
        self.pool = multiprocessing.Pool()

    def calc_heatmap(self, annotation_type, size):

        annotation_type = annotation_type.lower()

        canvas = np.zeros((size,size), dtype=np.float64)

        annotations = None
        if self.annotation_path.endswith('yaml') or self.annotation_path.endswith('yml'):
            with open(self.annotation_path, 'r') as f:
                annotations = yaml.safe_load(f)
        elif self.annotation_path.endswith('pkl'):
            with open(self.annotation_path, 'rb') as f:
                annotations = pickle.load(f)
        else:
            print("Unknown file type")

        for root,_,f_names in os.walk(self.image_path):
            f_names = sorted([f for f in f_names if f.endswith(".png") or f.endswith(".jpg")])
            if not annotation_type == 'lines':
                out_list = self.pool.map(
                    functools.partial(
                        render,
                        root,
                        annotations['images'],
                        annotation_type,
                        size,
                        canvas),
                    f_names)
            else:
                out_list = self.pool.map(
                    functools.partial(
                        render_lines,
                        self.lines_path,
                        canvas),
                    f_names)

            canvas += np.array(out_list).sum(axis=0)
        return canvas / len(f_names)

    def main(self):
        classes = ['Ball', 'Goalpost', 'Robot', 'Field Edge', 'T-Intersection', 'L-Intersection', 'X-Intersection', 'Lines']
        size = 100

        plt.figure(figsize=(8,3))
        sns.set_style("whitegrid")
        sns.set_context("paper")
        plt.rcParams["font.sans-serif"] = "arial"
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        for idx, cls in enumerate(classes):
            sub = plt.subplot(2, 4, idx + 1, xticks=[], yticks=[])
            heatmap = self.calc_heatmap(cls, size)
            vmax = 0.03
            if cls == 'Field Edge':
                vmax = 1.0
                cls = "Field Area"
            if cls == 'Robot':
                vmax = 0.1
            if cls == 'Ball':
                vmax = 0.06
            if cls == 'Lines':
                vmax = 0.02
            sub.set_title(f'{cls}')
            sns.heatmap(heatmap, xticklabels=False, yticklabels=False, linewidths=0.0, rasterized=True, vmin=0.0, vmax=vmax)

        plt.tight_layout()
        plt.savefig("heatmaps.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.show()


if __name__ == "__main__":
    image_path = "/home/jan/Schreibtisch/imageset1069/"
    annotation_path = "/home/jan/Downloads/vision_dataset_2021_labels.pkl"
    lines_path ="/home/jan/Schreibtisch/lines_out/"
    Heatmapper(image_path, annotation_path, lines_path).main()
