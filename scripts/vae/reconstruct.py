#!/bin/python

# Based on https://github.com/noctrog/conv-vae

import argparse

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import model

def normalize(img, mean, std):
    return (img - mean) / std

def unnormalize(img, mean, std):
    return img * std + mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help='input image')
    parser.add_argument("-w", type=str, default='', help='Model weights')
    parser.add_argument("--data_dir", help="Folder where the data is located")

    args = parser.parse_args()

    assert isinstance(args.i, str)
    assert isinstance(args.w, str)

    # load model
    dump = torch.load(args.w)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).cuda()
    vae.load_state_dict(dump['state_dict'])
    vae.eval()

    # load image
    img = np.asarray(Image.open(args.i).resize((112, 128))) / 255
    # img = np.asarray(Image.open(args.i)) / 255
    img = np.transpose(img, [2, 0, 1])
    img_v = torch.tensor(img, dtype=torch.float32).unsqueeze(0).cuda()
    img_v = torch.cat((img_v, img_v, img_v, img_v, img_v), 0)
    _, __, output_v = vae.forward(img_v)
    out_img = output_v.detach().squeeze(0).cpu().numpy()

    # plot
    fig = plt.figure()
    plt.subplot(3, 3, 1, xticks=[], yticks=[])
    plt.imshow(np.transpose(img, [1, 2, 0]))
    plt.subplot(3, 3, 2, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[0], [1, 2, 0]))
    plt.subplot(3, 3, 3, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[1], [1, 2, 0]))
    plt.subplot(3, 3, 4, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[2], [1, 2, 0]))
    plt.subplot(3, 3, 5, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[3], [1, 2, 0]))
    plt.subplot(3, 3, 6, xticks=[], yticks=[])
    plt.imshow(np.transpose(out_img[4], [1, 2, 0]))
    plt.show()


if __name__ == "__main__":
    main()
