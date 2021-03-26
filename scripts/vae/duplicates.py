#!/bin/python

# Based on https://github.com/noctrog/conv-vae

import argparse

import numpy as np
import pickle
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help='Input embeddings')
    parser.add_argument("-d", type=int, default=30, help='The distance in latent space arroud the sample that is pruned.')

    args = parser.parse_args()

    with open(args.i, "rb") as f:
        out_dict = pickle.load(f)

    path_list = out_dict['path_list']
    latent_spaces_numpy = out_dict['latent']
    tree = out_dict['tree']
    errors = out_dict['errors']

    error_threshold = errors.mean() + errors.std() * 1.64

    # Add all ones where the autoencoder performed bad without checking for the density
    not_recreatable = set([path_list[i] for i in np.where(errors >= error_threshold)[0]])
    
    print(f"{len(not_recreatable)} images are included due to their high error in the autoencoder.")

    finished_set = not_recreatable.copy()

    path_set = set(path_list)

    while len(path_set) > 0:

        print(len(path_set))

        element = path_set.pop()

        finished_set.add(element)

        idx = path_list.index(element)

        indices = tree.query_radius(
            latent_spaces_numpy[idx].reshape(1,-1), r=args.d)[0]

        for index in indices:
            if path_list[index] in path_set:
                path_set.remove(path_list[index])

    print(f"{len(finished_set)} images are included after the latent space distance sampling.")

    with open("selection.yaml", "w") as f:
        yaml.dump(
            {
                'high_autoencoder_error': list(not_recreatable), 
                'selection': list(finished_set),
                'dropout': list(set(path_list) - finished_set),
            }, f)

    fig = plt.figure()
    for idx, path in enumerate(sorted(list(finished_set))[0:50]):
        img = np.asarray(Image.open(path)) / 255
        sub = plt.subplot(5, 10, idx + 1, xticks=[], yticks=[])
        #sub.set_title(f'Distance: {dist[idx]:.2f}')
        plt.imshow(img)
    plt.show()

    """
        print(dist, ind, path)
        if idx == len(path_list)-2:
            fig = plt.figure()
            for idx, i in enumerate(ind):
                img = np.asarray(Image.open(path_list[i])) / 255
                sub = plt.subplot(3, 5, idx + 1, xticks=[], yticks=[])
                sub.set_title(f'Distance: {dist[idx]:.2f}')
                plt.imshow(img)
            plt.show()
    """


if __name__ == "__main__":
    main()
