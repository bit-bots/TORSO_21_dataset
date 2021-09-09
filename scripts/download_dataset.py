#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile
from urllib.parse import urlencode
from urllib.request import urlretrieve

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
DOWNLOAD_LINK = 'https://cloud.crossmodal-learning.org/s/3wt3Sgyxc7pC5QT'


def download(filename, params, folder, approx_size):
    print(f'Downloading dataset to {folder}... '
          f'This might take a lot of time and take up to {approx_size} GB of disk space')
    os.makedirs(folder, exist_ok=True)
    query = urlencode(params)
    urlretrieve(DOWNLOAD_LINK + '/download?' + query, filename)
    print('Download finished, extracting data...')
    with zipfile.ZipFile(filename) as f:
        f.extractall(folder)
    os.remove(filename)
    print('Extraction finished.')


def download_file(filename, params):
    print(f'Downloading annotations file to {filename}...')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    query = urlencode(params)
    urlretrieve(DOWNLOAD_LINK + '/download?' + query, filename)
    print('Download finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the TORSO-21 dataset.')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('-a', '--all', action='store_true', help='Download the complete dataset')
    grp.add_argument('-r', '--real', action='store_true', help='Download the images captured in real environments')
    grp.add_argument('-s', '--simulation', action='store_true', help='Download the images generated from simulation')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--test', action='store_true', help='Only download test data')
    grp.add_argument('--train', action='store_true', help='Only download training data')
    parser.add_argument('--annotations', action='store_true', help='Only download annotations')
    args = parser.parse_args()

    if not any(getattr(args, arg) for arg in args.__dict__):
        parser.error('Please specify which data to download. Use --help for further information.')

    if args.all and any((args.test, args.train, args.annotations)):
        parser.error('--all cannot be used with other options.')

    if args.all:
        tmp_file = os.path.join(DATA_FOLDER, 'dataset.zip')
        params = {}
        folder = DATA_FOLDER
        approx_size = 190
        download(tmp_file, params, folder, approx_size)
    elif args.real and args.test and args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'reality', 'test', 'annotations.yaml'),
                      {'path': '/reality/test', 'files': 'annotations.yaml'})
    elif args.real and args.test:
        tmp_file = os.path.join(DATA_FOLDER, 'reality', 'test.zip')
        params = {
            'path': '/reality',
            'files': 'test',
        }
        folder = os.path.join(DATA_FOLDER, 'reality')
        approx_size = 2
        download(tmp_file, params, folder, approx_size)
    elif args.real and args.train and args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'reality', 'train', 'annotations.yaml'),
                      {'path': '/reality/train', 'files': 'annotations.yaml'})
    elif args.real and args.train:
        tmp_file = os.path.join(DATA_FOLDER, 'reality', 'train.zip')
        params = {
            'path': '/reality',
            'files': 'train',
        }
        folder = os.path.join(DATA_FOLDER, 'reality')
        approx_size = 10
        download(tmp_file, params, folder, approx_size)
    elif args.real and args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'reality', 'train', 'annotations.yaml'),
                      {'path': '/reality/train', 'files': 'annotations.yaml'})
        download_file(os.path.join(DATA_FOLDER, 'reality', 'test', 'annotations.yaml'),
                      {'path': '/reality/test', 'files': 'annotations.yaml'})
    elif args.real:
        tmp_file = os.path.join(DATA_FOLDER, 'reality.zip')
        params = {
            'path': '/',
            'files': 'reality',
        }
        folder = DATA_FOLDER
        approx_size = 11
        download(tmp_file, params, folder, approx_size)
    elif args.simulation and args.test and args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'simulation', 'test', 'annotations.yaml'),
                      {'path': '/simulation/test', 'files': 'annotations.yaml'})
    elif args.simulation and args.test:
        tmp_file = os.path.join(DATA_FOLDER, 'simulation', 'test.zip')
        params = {
            'path': '/simulation',
            'files': 'test',
        }
        folder = os.path.join(DATA_FOLDER, 'simulation')
        approx_size = 28
        download(tmp_file, params, folder, approx_size)
    elif args.simulation and args.train and args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'simulation', 'train', 'annotations.yaml'),
                      {'path': '/simulation/train', 'files': 'annotations.yaml'})
    elif args.simulation and args.train:
        tmp_file = os.path.join(DATA_FOLDER, 'simulation', 'train.zip')
        params = {
            'path': '/simulation',
            'files': 'train',
        }
        folder = os.path.join(DATA_FOLDER, 'simulation')
        approx_size = 152
        download(tmp_file, params, folder, approx_size)
    elif args.simulation and args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'simulation', 'train', 'annotations.yaml'),
                      {'path': '/simulation/train', 'files': 'annotations.yaml'})
        download_file(os.path.join(DATA_FOLDER, 'simulation', 'test', 'annotations.yaml'),
                      {'path': '/simulation/test', 'files': 'annotations.yaml'})
    elif args.simulation:
        tmp_file = os.path.join(DATA_FOLDER, 'simulation.zip')
        params = {
            'path': '/',
            'files': 'simulation',
        }
        folder = DATA_FOLDER
        approx_size = 180
        download(tmp_file, params, folder, approx_size)
    elif args.annotations:
        download_file(os.path.join(DATA_FOLDER, 'reality', 'train', 'annotations.yaml'),
                      {'path': '/reality/train', 'files': 'annotations.yaml'})
        download_file(os.path.join(DATA_FOLDER, 'reality', 'test', 'annotations.yaml'),
                      {'path': '/reality/test', 'files': 'annotations.yaml'})
        download_file(os.path.join(DATA_FOLDER, 'simulation', 'train', 'annotations.yaml'),
                      {'path': '/simulation/train', 'files': 'annotations.yaml'})
        download_file(os.path.join(DATA_FOLDER, 'simulation', 'test', 'annotations.yaml'),
                      {'path': '/simulation/test', 'files': 'annotations.yaml'})
