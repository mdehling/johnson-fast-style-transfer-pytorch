import torch

from torchvision.models.vgg import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Normalize


def total_variation(x):
    tv_h = (x[...,1:,:] - x[...,:-1,:]).abs().sum()
    tv_w = (x[...,:,1:] - x[...,:,:-1]).abs().sum()
    return tv_h + tv_w


def avg_gram(x):
    return torch.einsum('brij,bsij->brs', x, x) / (x.shape[2]*x.shape[3])


class GatysLoss:

    def __init__(self,
        content_image=None,
        style_image=None,
        content_weight=1.0,
        style_weight=0.3,
        var_weight=1e-5,
        device=None
        ):
        """Gatys Loss."""

        self.content_layers = ['features.15']
        self.style_layers = [
            'features.3', 'features.8', 'features.15', 'features.22'
        ]
        self.feature_model = create_feature_extractor(
            vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES),
            self.content_layers + self.style_layers
        ).eval().requires_grad_(False).to(device=device)

        self.device = device

        self.normalize = Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        if content_image is not None:
            content_image = self.normalize(content_image)
            content_features = self.feature_model(content_image)
            self.content_features = {
                layer: content_features[layer]
                for layer in self.content_layers
            }
        else:
            self.content_features = None

        if style_image is not None:
            style_image = self.normalize(style_image)
            style_features = self.feature_model(style_image)
            self.style_features = {
                layer: avg_gram(style_features[layer])
                for layer in self.style_layers
            }
        else:
            self.style_features = None

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.var_weight = var_weight

    def __call__(self, pastiche_image, content_image=None, style_image=None):
        pastiche_image = self.normalize(pastiche_image)
        pastiche_features = self.feature_model(pastiche_image)
        pastiche_content_features = {
            layer: pastiche_features[layer]
            for layer in self.content_layers
        }
        pastiche_style_features = {
            layer: avg_gram(pastiche_features[layer])
            for layer in self.style_layers
        }

        if self.content_features is not None:
            content_features = self.content_features
        else:
            content_image = self.normalize(content_image)
            content_features = self.feature_model(content_image)

        if self.style_features is not None:
            style_features = self.style_features
        else:
            style_image = self.normalize(style_image)
            style_features = self.feature_model(style_image)
            style_features = {
                layer: avg_gram(style_features[layer])
                for layer in self.style_layers
            }

        content_loss = sum([
            torch.mean(torch.square(
                pastiche_content_features[layer] - content_features[layer]
            ))
            for layer in self.content_layers
        ])

        style_loss = sum([
            torch.mean(torch.square(
                pastiche_style_features[layer] - style_features[layer]
            ))
            for layer in self.style_layers
        ])

        var_loss = total_variation(pastiche_image)

        return (
            self.content_weight * content_loss
            + self.style_weight * style_loss
            + self.var_weight * var_loss
        )

