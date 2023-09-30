import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Normalize
from torchvision.models.vgg import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class ConvBlock(nn.Module):

    def __init__(self,
        in_features,
        out_features,
        kernel_size,
        stride=1,
        transpose=False,
        normalization='batch',
        activation='relu',
        **kwargs):
        """Convolutional Block."""
        super().__init__(**kwargs)

        if not transpose:
            self.conv = nn.Conv2d(
                in_features, out_features, kernel_size,
                stride=stride, padding=(kernel_size-1)//2
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_features, out_features, kernel_size,
                stride=stride, padding=(kernel_size-1)//2, output_padding=1
            )

        match normalization:
            case 'batch':       self.norm = nn.BatchNorm2d(out_features)
            case 'instance':    self.norm = nn.InstanceNorm2d(out_features)
            case None:          self.norm = None
            case _:             raise ValueError()

        match activation:
            case 'relu':        self.act = nn.ReLU()
            case 'tanh':        self.act = nn.Tanh()
            case None:          self.act = None
            case _:             raise ValueError()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, features, normalization='batch', **kwargs):
        """Residual Block."""
        super().__init__(**kwargs)

        self.conv1 = nn.Conv2d(features, features, 3, stride=1)
        match normalization:
            case 'batch':       self.norm1 = nn.BatchNorm2d(features)
            case 'instance':    self.norm1 = nn.InstanceNorm2d(features)
            case _:             raise ValueError()
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(features, features, 3, stride=1)
        match normalization:
            case 'batch':       self.norm2 = nn.BatchNorm2d(features)
            case 'instance':    self.norm2 = nn.InstanceNorm2d(features)
            case _:             raise ValueError()

        self.res = nn.ZeroPad2d(-2)

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))
        return self.norm2(self.conv2(y)) + self.res(x)


class PreProcessing(nn.Module):

    def __init__(self, **kwargs):
        """Pre Processing."""
        super().__init__()
        self.normalize = Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

    def forward(self, x):
        return self.normalize(x)


class PostProcessing(nn.Module):

    def __init__(self, **kwargs):
        """Post Processing."""
        super().__init__()

    def forward(self, x):
        mean = torch.as_tensor(
            (0.485,0.456,0.406),
            dtype=x.dtype, device=x.device
        ).view(-1,1,1)
        return torch.clamp(x*0.6 + mean, 0.0, 1.0)


class JohnsonStyleTransfer(nn.Module):

    def __init__(self,
        filters=(32,64,128),
        normalization='batch',
        **kwargs):
        """Johnson Style Transfer Model."""
        super().__init__(**kwargs)

        self.prep = PreProcessing()
        self.rpad = nn.ReflectionPad2d(40)

        self.conv_block_1 = ConvBlock(3, filters[0], 9, 1,
            normalization=normalization)
        self.conv_block_2 = ConvBlock(filters[0], filters[1], 3, 2,
            normalization=normalization)
        self.conv_block_3 = ConvBlock(filters[1], filters[2], 3, 2,
            normalization=normalization)

        self.res_block_1 = ResidualBlock(filters[2],
            normalization=normalization)
        self.res_block_2 = ResidualBlock(filters[2],
            normalization=normalization)
        self.res_block_3 = ResidualBlock(filters[2],
            normalization=normalization)
        self.res_block_4 = ResidualBlock(filters[2],
            normalization=normalization)
        self.res_block_5 = ResidualBlock(filters[2],
            normalization=normalization)

        self.conv_block_4 = ConvBlock(filters[2], filters[1], 3, 2,
            transpose=True, normalization=normalization)
        self.conv_block_5 = ConvBlock(filters[1], filters[0], 3, 2,
            transpose=True, normalization=normalization)
        self.conv_block_6 = ConvBlock(filters[0], 3, 9, 1,
            normalization=None, activation='tanh')

        self.post = PostProcessing()

    def forward(self, x):
        x = self.prep(x)
        x = self.rpad(x)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)

        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)

        return self.post(x)

