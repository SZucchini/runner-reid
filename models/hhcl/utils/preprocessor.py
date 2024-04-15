import torch
from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        imgs = []
        imgs_path, pid, camid = self.dataset[idx]
        fname = imgs_path[0]

        for p in imgs_path:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)

        return imgs, fname, pid, camid, idx
