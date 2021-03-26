#!/bin/python

# Based on https://github.com/noctrog/conv-vae

import argparse

import numpy as np
import pickle
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from torchvision import datasets, transforms, utils

from tqdm import tqdm

import model


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


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

    data_set = ImageFolderWithPaths(args.data_dir, transform=train_transforms)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    dump = torch.load(args.w)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).cuda()
    vae.load_state_dict(dump['state_dict'])
    vae.eval()

    path_list = list()

    latent_spaces_numpy = np.empty((len(data_set), dump['z_dim']))

    error_function = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (images, _, paths) in enumerate(tqdm(data_loader)):
            images_v = images.to("cuda")
            mu, out_img, _ = vae(images_v, sampling=False)
            mu = mu.to("cpu")
            for img_idx, path in enumerate(paths):
                error = error_function(out_img[img_idx], images_v[img_idx]).detach().to("cpu")
                print(error)
                np_mu = mu[img_idx].detach().numpy()
                latent_spaces_numpy[img_idx + (batch_idx * args.batch_size)] = np_mu
                path_list.append(path)

    print(len(path_list), latent_spaces_numpy.shape[0])

    print("Build Tree!")

    tree = KDTree(latent_spaces_numpy)

    out_dict = {
        'path_list': path_list,
        'latent': latent_spaces_numpy,
        'tree': tree,
    }

    print("Save embeddings!")

    with open('embeddings.pickle', 'wb') as f:
        pickle.dump(out_dict, f)

    print("Finished!")


if __name__ == "__main__":
    main()
