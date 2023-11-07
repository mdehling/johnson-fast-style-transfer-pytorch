import os
import torch

dependencies = [
    'torch',
    'torchvision',
]

def johnson2016(style):
    """
    Johnson et al (2016) Fast Style Transfer.

    Args:
        style:
            A string indicating which pre-trained model to load.  One of:
            `'bathing'`, `'candy'`, `'cubism'`, `'delaunay'`, `'scream'`,
            `'starry-night'`, or `'udnie'`.

    Returns:
        A pre-trained model for the indicated style.  The model takes a tensor
        of shape `(B,C,H,W)` representing a batch of content images and returns
        a tensor of the same shape representing the generated pastiche images.

    References:
        * Johnson, Alahi, Fei-Fei.  "Perceptual Losses for Real-Time Style
            Transfer and Super-Resolution."  ECCV, 2016.
        * Ulyanov, Vedaldi, Lempitsky.  "Instance Normalization: The Missing
            Ingredient for Fast Stylization."  Arxiv, 2016.
    """
    # FIXME: I should really host the model weights separately and use
    # torch.hub.load_state_dict_from_url() to load them.  Unfortunately using
    # google drive for hosting doesn't work.
    model_path = os.path.join(
        torch.hub.get_dir(),
        "mdehling_johnson-fast-style-transfer-pytorch_main",
        f"multirun/2023-10-01/19-39-01/style_image={style}.jpg/model.pth"
    )
    return torch.load(model_path)
