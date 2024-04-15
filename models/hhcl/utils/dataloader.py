import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader
from torchvision import transforms

from .preprocessor import Preprocessor
from .sampler import RandomMultipleGallerySampler


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def unified_seqlen(batch, cfg):
    imgs, fname, pid, camid, idx = list(zip(*batch))
    # covert list to tensor
    pid = torch.tensor(pid)
    camid = torch.tensor(camid)
    idx = torch.tensor(idx)

    images_flipped = [img.flip(0) for img in imgs]
    pad_len = cfg.INPUT.MAX_SEQ_LEN - images_flipped[0].shape[0]
    if pad_len > 0:
        pad_tensor = torch.zeros(pad_len, 3, cfg.INPUT.IMAGE_SIZE[0], cfg.INPUT.IMAGE_SIZE[1])
        images_flipped[0] = torch.cat([images_flipped[0], pad_tensor], dim=0)

    inputs = rnn.pad_sequence(images_flipped, batch_first=True, padding_value=0)
    inputs = inputs.flip(1)
    del imgs, images_flipped
    return inputs, fname, pid, camid, idx


def collate_wrapper(cfg):
    def collate(batch):
        return unified_seqlen(batch, cfg)
    return collate


def get_test_loader(cfg, dataset, testset=None):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    test_transformer = transforms.Compose([
        transforms.Resize(cfg.INPUT.IMAGE_SIZE, interpolation=3),
        transforms.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    collate_fn = collate_wrapper(cfg)
    test_loader = DataLoader(
        Preprocessor(testset, transform=test_transformer),
        batch_size=cfg.INPUT.BATCH_SIZE, num_workers=cfg.INPUT.WORKERS,
        shuffle=False, pin_memory=True, collate_fn=collate_fn)

    return test_loader


def get_train_loader(cfg, dataset, trainset=None):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_transformer = transforms.Compose([
        transforms.Resize(cfg.INPUT.IMAGE_SIZE, interpolation=3),
        transforms.ToTensor(),
        normalizer,
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = cfg.INPUT.INSTANCES > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, cfg.INPUT.INSTANCES)
    else:
        sampler = None

    collate_fn = collate_wrapper(cfg)
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, transform=train_transformer),
                   batch_size=cfg.INPUT.BATCH_SIZE, num_workers=cfg.INPUT.WORKERS, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True, collate_fn=collate_fn),
        length=cfg.TRAIN.ITERS)

    return train_loader
