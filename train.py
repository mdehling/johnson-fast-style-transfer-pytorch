#!/usr/bin/env python

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

from models import JohnsonStyleTransfer
from losses import GatysLoss
from datasets import COCO2014


def total_variation(x):
    tv_h = (x[...,1:,:] - x[...,:-1,:]).abs().sum()
    tv_w = (x[...,:,1:] - x[...,:,:-1]).abs().sum()
    return tv_h + tv_w


class JohnsonStyleTransferTrainer:
    # When saving & loading this class, we are missing one vital piece of
    # information: the batch size.  Since a larger batch size has a
    # regularizing effect, it is an important training parameter.

    def __init__(self, model, style_image,
        content_weight=1.0,
        style_weight=5.0,
        var_weight=1e-5,
        lr=1e-3,
        device=None):
        """Johnson Style Transfer Model Trainer."""
        self.model = model.train()
        self.style_image = style_image

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.var_weight = var_weight

        self.gatys_loss = GatysLoss(
            style_image = style_image,
            content_weight = content_weight,
            style_weight = style_weight,
            device = device
        )
        self.optimizer = Adam(model.parameters(), lr=lr)

        self.device = device

    def __call__(self, dl):
        for content_image in dl:
            content_image = content_image.to(device=self.device)

            pastiche_image = self.model(content_image)

            total_loss = self.gatys_loss(pastiche_image, content_image)
            total_loss += self.var_weight * total_variation(pastiche_image)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


def parse_args():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Train Johnson (2016) style transfer model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('style_image',
                        help='style image file')

    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='content weight')
    parser.add_argument('--style_weight', type=float, default=3e-1,
                        help='style weight')
    parser.add_argument('--var_weight', type=float, default=1e-5,
                        help='variation weight')
    parser.add_argument('--normalization',
                        choices=['batch', 'instance'], default='instance',
                        help='type of normalization' )
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of data loading worker threads')

    parser.add_argument('--data_dir', default='coco2014',
                        help='dataset directory')
    parser.add_argument('--saved_model',
                        help='''where to save the trained model.
                        If not provided, the model is saved in 'saved/{stem}',
                        i.e., given a style_image 'img/style/candy.jpg', the
                        trained model is written to 'saved/candy.pt'.''')

    args = parser.parse_args()

    if args.saved_model is None:
        saved_model = Path('saved') / Path(args.style_image).name.with_suffix('.pt')
        args.saved_model = str(saved_model)

    return args


if __name__ == '__main__':
    args = parse_args()

    ds = COCO2014(args.data_dir)
    dl = DataLoader(ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    style_image = Image.open(args.style_image)
    style_image = transforms.functional.to_tensor(style_image)
    style_image = torch.unsqueeze(style_image, 0)
    style_image = style_image.to(device=device)

    model = JohnsonStyleTransfer(normalization=args.normalization)
    model = model.train().to(device=device)

    trainer = JohnsonStyleTransferTrainer(model, style_image,
        content_weight = args.content_weight,
        style_weight = args.style_weight,
        var_weight = args.var_weight,
        lr = args.learning_rate,
        device = device
    )

    for i in range(args.epochs):
        print(f"epoch {i+1}")
        trainer(tqdm(dl))

    model.eval().to(device='cpu')
    torch.save(model, args.saved_model)
