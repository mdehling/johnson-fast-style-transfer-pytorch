import os
import torch

dependencies = [
    'torch',
    'torchvision',
]

def johnson2016(style):
    # FIXME: I should really host the model weights separately and use
    # torch.hub.load_state_dict_from_url() to load them.  Unfortunately using
    # google drive for hosting doesn't work.
    model_path = os.path.join(
        torch.hub.get_dir(),
        "mdehling_johnson-fast-style-transfer-pytorch_main",
        f"multirun/2023-10-01/19-39-01/style_image={style}.jpg/model.pth"
    )
    return torch.load(model_path)
