from dataclasses import dataclass, field
from typing import Any, List, Tuple

from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class NetworkConfig:
    filters: Tuple[int] = (32,64,128)
    normalization: str = 'instance'
    upsampling: str = 'conv'


@dataclass
class DataConfig:
    path: str = 'coco2014'
    batch_size: int = 32
    num_workers: int = 6


@dataclass
class LossConfig:
    # backend: str = vgg16 (preprocess)
    content_weight: float = 1.0
    style_weight: float = 0.3
    var_weight: float = 1e-5


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    epochs: int = 2


@dataclass
class Config:
    content_image_dir: str = 'img/content'
    style_image_dir: str = 'img/style'
    style_image: str = MISSING

    save_model: str = 'model.pth'
    save_state: str = 'state.pth'

    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    hydra: Any = field(default_factory=lambda: {
        'sweep': { 'subdir': '${hydra:job.override_dirname}' }
    })


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
