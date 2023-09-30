#!/usr/bin/env python

import argparse

import torch

from nst.io import load_image, save_image


def parse_args():

    parser = argparse.ArgumentParser(
        description='Stylize using Johnson (2016) style transfer model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('model',
                        help='saved model file')
    parser.add_argument('content_image',
                        help='content image file')
    parser.add_argument('pastiche_image',
                        help='pastiche (output) image file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    model = torch.load(args.model).eval().to(device='cpu')

    content_image = load_image(args.content_image)
    pastiche_image = model(content_image)
    save_image(args.pastiche_image, pastiche_image)
