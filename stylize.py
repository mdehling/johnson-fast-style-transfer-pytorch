#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image


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

    content_image = Image.open(args.content_image)
    content_image = transforms.functional.to_tensor(content_image)
    content_image = torch.unsqueeze(content_image, 0)

    pastiche_image = model(content_image)

    pastiche_image = torch.squeeze(pastiche_image, 0)
    pastiche_image = transforms.functional.to_pil_image(pastiche_image)
    pastiche_image.save(args.pastiche_image)
