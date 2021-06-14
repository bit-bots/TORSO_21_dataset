#!/usr/bin/env python3
import os
import numpy as np
import cv2
import time
import pickle
import yaml
from tqdm import tqdm


class SegmentationMerge(object):
    def __init__(self):
        self.img_path="/home/florian/Downloads/imageset/"
        self.annotation_path="/home/florian/Downloads/vision_dataset_2021_labels(2).yaml"
        self.mask_path = "/home/florian/Projekt/bitbots/vision_dataset_2021/masks/"
        self.seg_out_path = "/home/florian/Projekt/bitbots/vision_dataset_2021/segmentations/"
        self.line_out_path = "/home/florian/Projekt/bitbots/vision_dataset_2021/masks_lines/"


    def main(self):
        annotations = None
        if self.annotation_path.endswith('yaml') or self.annotation_path.endswith('yml'):
            with open(self.annotation_path, 'r') as f:
                annotations = yaml.safe_load(f)
        elif self.annotation_path.endswith('pkl'):
            with open(self.annotation_path, 'rb') as f:
                annotations = pickle.load(f)
        else:
            print("Unknown file type")

        annotations = annotations['images']

        for root,_,f_names in os.walk(self.img_path):

            f_names = sorted([f for f in f_names if f.endswith(".png") or f.endswith(".jpg")])

            for f in tqdm(f_names):
                mask_name = os.path.splitext(os.path.basename(f))[0] + '.png'
                mask = cv2.imread(os.path.join(self.mask_path, mask_name))
                if mask is None:
                    continue

                # Check shapes as some masks may be smaller due to old bug in line label tool or due to being labeled by yolo
                image = cv2.imread(os.path.join(root, f))
                if image.shape != mask.shape:
                    print(f"Shape missmatch {mask.shape} != {image.shape} on: '{f}'")
                    if image.shape[0] - 10 == mask.shape[0] and image.shape[1] - 10 == mask.shape[1]:  # Caused by bug, pad masks  # TODO check for offset
                        print(f"Padding: '{f}'")
                        mask = np.pad(mask, (5,5))[:,:,5:-5]
                    else:  # Yolo output, resize to original shape
                        print(f"Resizing from {mask.shape} to {image.shape}: '{f}'")
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                if mask.shape != image.shape:
                    print(f"Something went wrong on '{f}'")

                mask[mask < 2] == 0

                shape = mask.shape

                annotations_of_type_in_image = filter(
                    lambda annotation:
                        annotation['type'] in ['field edge', 'ball', "goalpost"] and annotation['in_image'],
                    annotations[f]["annotations"])

                field = None

                for annotation in annotations_of_type_in_image:

                    vector = np.array(annotation['vector'], dtype = np.int32)

                    if annotation['type'] == 'field edge':
                        if vector[0,0] < vector[-1,0]:
                            vector = np.array([[0,shape[0]]] + vector.tolist() + [[shape[1], shape[0]]]).astype(np.int32)
                        else:
                            vector = np.array([[shape[1], shape[0]]] + vector.tolist() + [[0,shape[0]]]).astype(np.int32)

                        # Sticky edges
                        for i in range(vector.shape[0]-1):
                            if vector[i, 1] < 20:
                                vector[i, 1] = 0
                            if vector[i, 1] > (shape[0] - 20):
                                vector[i, 1] = shape[0]

                        field = cv2.fillConvexPoly(np.zeros_like(mask), vector, (255, 255, 255)) // 255

                        mask *= field

                    elif annotation['type'] == 'ball':
                        mask = cv2.circle(
                            mask.astype(np.int32),
                            (
                                (vector[0][0] + vector[1][0]) // 2,
                                (vector[0][1] + vector[1][1]) // 2
                            ),
                            ((vector[1][0] - vector[0][0]) + (vector[1][1] - vector[0][1])) // 4,
                            (0,0,0), -1)
                    elif annotation['type'] == 'goalpost':
                        mask = cv2.fillConvexPoly(mask.astype(np.int32), vector, (0, 0, 0))

                if field is None:
                    print(f"skipped image without fieldboundary: '{f}'")
                    continue

                seg = (field * (255 - mask)) + mask // 2

                cv2.imwrite(os.path.join(self.seg_out_path, mask_name), seg)
                cv2.imwrite(os.path.join(self.line_out_path, mask_name), mask // 255)


if __name__ == "__main__":
    SegmentationMerge().main()
