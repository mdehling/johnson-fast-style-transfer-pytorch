import torch
from torchvision import datasets, transforms


class COCO2014(torch.utils.data.Dataset):

    def __init__(self, root):
        super().__init__()
        self.labeled_images = datasets.ImageFolder(root,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256, antialias=False),    # resize short side
                transforms.RandomCrop(256)                  # crop to square
            ])
        )

    def __len__(self):
        return len(self.labeled_images)

    def __getitem__(self, idx):
        image, label = self.labeled_images[idx]
        return image
