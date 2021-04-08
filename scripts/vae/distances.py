#!/bin/python

# Based on https://github.com/noctrog/conv-vae

import os
import argparse

import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=str, help='Input embeddings')
    parser.add_argument("-i", type=str, help='Image which neigbours are calculated')

    args = parser.parse_args()

    with open(args.e, "rb") as f:
        out_dict = pickle.load(f)

    path_list = out_dict['path_list']
    latent_spaces_numpy = out_dict['latent']
    tree = out_dict['tree']
    errors = out_dict['errors']

    idx = path_list.index(args.i)

    distances, indices = tree.query(
            latent_spaces_numpy[idx].reshape(1,-1), k=10000)

    print(indices)

    fig = plt.figure(figsize=(4,3))
    
    for idx, iidx in enumerate([0,1,35,9999]):
        img = np.asarray(Image.open(path_list[indices[0, iidx]])) / 255
        sub = plt.subplot(2, 2, idx + 1, xticks=[], yticks=[])
        sub.set_title(f'N={iidx} D={distances[0,iidx]:.2f}')
        plt.imshow(img)

    plt.subplots_adjust(hspace=0.3)

    plt.savefig("image_dist_fig.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)
    plt.show()


if __name__ == "__main__":
    main()
