#!/bin/python

# Based on https://github.com/noctrog/conv-vae

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter

import argparse

import model

def kl_loss(mu, log_var):
    # TODO: dividir entre el numero de batches?
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.exp(log_var))

def r_loss(y_train, y_pred):
    r_loss = torch.mean((y_train - y_pred) ** 2)
    return r_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--data_dir", help="Folder where the data is located")
    parser.add_argument("--epochs", type=int, help='Number of times to iterate the whole dataset')
    parser.add_argument("--visual_every", default=10, type=int, help='Display faces every n batches')
    parser.add_argument("--z_dim", type=int, default=200, help='Dimensions of latent space')
    parser.add_argument("--r_loss_factor", type=float, default=10000.0, help='r_loss factor')
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size')
    parser.add_argument("--load", type=str, default='', help='Load pretrained weights')
    args = parser.parse_args()

    # data where the images are located
    data_dir = args.data_dir
    assert isinstance(data_dir, str)
    assert isinstance(args.epochs, int)
    assert isinstance(args.visual_every, int)
    assert isinstance(args.z_dim, int)
    assert isinstance(args.r_loss_factor, float)
    assert isinstance(args.lr, float)
    assert isinstance(args.batch_size, int)

    # use CPU or GPU
    device = torch.device("cuda" if args.cuda else "cpu")

    # prepare data
    train_transforms = transforms.Compose([
        # transforms.Resize((176, 144)),
        transforms.Resize((128, 112)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    images, labels = next(iter(trainloader))

    # create model
    input_shape = next(iter(trainloader))[0].shape
    vae = model.VAE(input_shape[-3:], args.z_dim).to(device)
    print(vae)          # print for feedback

    # load previous weights (if any)
    if args.load is not '':
        vae.load_state_dict(torch.load(args.load)['state_dict'])
        print("Weights loaded: {}".format(args.load))

    # create tensorboard writer
    writer = SummaryWriter(comment='-' + 'VAE' + str(args.z_dim))

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    # generate random points in latent space so we can see how the network is training
    latent_space_test_points = np.random.normal(scale=1.0, size=(16, args.z_dim))
    latent_space_test_points_v = torch.Tensor(latent_space_test_points).to(device)

    batch_iterations = 0
    training_losses = []
    vae.train()
    for e in range(args.epochs):
        epoch_loss = []
        for images, labels in tqdm(trainloader):
            images_v = images.to(device)

            optimizer.zero_grad()

            mu_v, log_var_v, images_out_v = vae(images_v)
            r_loss_v = r_loss(images_out_v, images_v)
            kl_loss_v = kl_loss(mu_v, log_var_v)
            loss = kl_loss_v + r_loss_v * args.r_loss_factor
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            if batch_iterations % args.visual_every == 0:
                # print loss
                print("Batch: {}\tLoss: {}".format(batch_iterations + e * len(trainloader) / args.batch_size, loss.item()))
                writer.add_scalar('loss', np.mean(epoch_loss[-args.visual_every:]), batch_iterations)


            batch_iterations = batch_iterations + 1

        else:
            training_losses.append(np.mean(epoch_loss))
            if min(training_losses) == training_losses[-1]:
                vae.save('vae-' + str(args.z_dim) + '.dat')

            vae.eval()

            generated_imgs_v = vae.forward_decoder(latent_space_test_points_v).detach()
            imgs_grid = utils.make_grid(generated_imgs_v)

            writer.add_image('preview-1', imgs_grid.cpu().numpy(), batch_iterations)

            vae.train()


if __name__ == "__main__":
    main()
