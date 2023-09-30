#!/usr/bin/env python

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from nst.config import Config
from nst.datasets import COCO2014
from nst.models import JohnsonStyleTransfer
from nst.losses import GatysLoss
from nst.io import load_image, save_image


@hydra.main(version_base='1.2', config_name='config')
def main(cfg: Config):
    hydra_cfg = HydraConfig.get()
    input_dir = Path(hydra_cfg.runtime.cwd)
    output_dir = Path(hydra_cfg.runtime.output_dir)

    ds = COCO2014(input_dir/cfg.training.data.path)
    dl = DataLoader(ds,
        batch_size=cfg.training.data.batch_size,
        shuffle=True,
        num_workers=cfg.training.data.num_workers,
        drop_last=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    style_image = load_image(
        input_dir / cfg.style_image_dir / cfg.style_image
    ).to(device=device)

    model = JohnsonStyleTransfer(
            filters=cfg.network.filters,
        normalization=cfg.network.normalization
    ).train().to(device=device)

    loss_fn = GatysLoss(
        style_image = style_image,
        content_weight = cfg.training.loss.content_weight,
        style_weight = cfg.training.loss.style_weight,
        var_weight = cfg.training.loss.var_weight,
        device = device
    )
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.optimizer.learning_rate
    )

    for i in range(cfg.training.epochs):
        print(f"epoch {i+1}")
        for content_image in tqdm(dl):
            content_image = content_image.to(device=device)
            pastiche_image = model(content_image)

            loss = loss_fn(pastiche_image, content_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval().to(device='cpu')
    torch.save(model, output_dir/cfg.save_model)

    if not OmegaConf.is_missing(cfg, 'save_state'):
        torch.save(optimizer.state_dict, output_dir/cfg.save_state)

    for filepath in (input_dir/cfg.content_image_dir).iterdir():
        try:
            content_image = load_image(filepath)
            pastiche_image = model(content_image)
            save_image(output_dir/filepath.name, pastiche_image)
        except:
            pass

if __name__ == '__main__':
    main()
