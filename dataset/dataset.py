import glob

import torch
import torch.nn.utils.rnn as rnn
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class RunnerDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = self._load_path()

    def _load_path(self):
        images_path = []
        for root in self.root_dirs:
            dirs = glob.glob(root + "/*")
            dirs = natsorted(dirs)
            for d in dirs:
                files = glob.glob(d + "/*.jpg")
                files = natsorted(files)
                images_path.append(files)
        return images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        images = []
        img_path = self.images_path[idx]
        for p in img_path:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images, dim=0)
        label = img_path[0].split("/")[-2]
        return images, label


def unified_seqlen(batch, cfg):
    images = []
    for img, _ in batch:
        images.append(img)

    images_flipped = [img.flip(0) for img in images]
    pad_len = cfg.INPUT.MAX_SEQ_LEN - images_flipped[0].shape[0]
    if pad_len > 0:
        pad_tensor = torch.zeros(pad_len, 3, cfg.INPUT.IMAGE_SIZE[0], cfg.INPUT.IMAGE_SIZE[1])
        images_flipped[0] = torch.cat([images_flipped[0], pad_tensor], dim=0)

    inputs = rnn.pad_sequence(images_flipped, batch_first=True, padding_value=0)
    inputs = inputs.flip(1)
    del images, images_flipped
    return inputs


def collate_wrapper(cfg):
    def collate(batch):
        return unified_seqlen(batch, cfg)
    return collate


def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    if cfg.DATASETS.TYPE == "NIGHT":
        root = [cfg.DATASETS.NIGHT_ROOT]
    elif cfg.DATASETS.TYPE == "DAYTIME":
        root = [cfg.DATASETS.DAYTIME_ROOT]

    dataset = RunnerDataset(root, transform=transform)
    train_length = int(len(dataset) * 0.8)
    valid_length = len(dataset) - train_length
    train_dataset, valid_dataset = random_split(dataset,
                                                [train_length, valid_length],
                                                generator=torch.Generator().manual_seed(cfg.SEED))

    collate_fn = collate_wrapper(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.INPUT.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.INPUT.BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader
