#!/bin/python

# Based on https://github.com/noctrog/conv-vae

import argparse
import os

import numpy as np
import pickle
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset

from tqdm import tqdm

import model
import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=str, default='', help='Model weights')
    parser.add_argument("--data_dir", help="Folder where the data is located")
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size')

    args = parser.parse_args()

    print("Starting...")

    # prepare data
    train_transforms = transforms.Compose([
        # transforms.Resize((176, 144)),
        transforms.Resize((128, 112)),
        transforms.ToTensor(),
    ])

    data_set = dataset.ImageFolderWithPaths(args.data_dir, transform=train_transforms)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    dump = torch.load(args.w)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).cuda()
    vae.load_state_dict(dump['state_dict'])
    vae.eval()

    path_list = list()

    latent_spaces_numpy = np.empty((len(data_set), dump['z_dim']))

    errors_numpy = np.empty((len(data_set)))

    error_function = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(tqdm(data_loader)):
            images_v = images.to("cuda")
            mu, _, out_img = vae(images_v, sampling=False)
            mu = mu.to("cpu")
            for img_idx, path in enumerate(paths):
                total_image_idx = img_idx + (batch_idx * args.batch_size)

                errors_numpy[total_image_idx] = error_function(
                        out_img[img_idx], 
                        images_v[img_idx]
                    ).detach().to("cpu")
                np_mu = mu[img_idx].detach().numpy()
                latent_spaces_numpy[total_image_idx] = np_mu
                path_list.append(path)

    print(len(path_list), latent_spaces_numpy.shape[0])

    print("Build Tree!")

    tree = KDTree(latent_spaces_numpy)

    out_dict = {
        'path_list': path_list,
        'latent': latent_spaces_numpy,
        'tree': tree,
        'errors': errors_numpy,
    }

    print("Save embeddings!")

    out_dir = os.path.join(args.data_dir, 'embeddings.pickle')

    with open(out_dir, 'wb') as f:
        pickle.dump(out_dict, f)

    print(f"Written outputs to '{out_dir}'.")

    print("Finished!")


if __name__ == "__main__":
    main()
