import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from PIL import Image


def load_image(image_file):
    image = Image.open(image_file)
    image = to_tensor(image)
    return image.unsqueeze(0)


def save_image(image_file, image):
    image = image.squeeze(0)
    image = to_pil_image(image)
    image.save(image_file)
