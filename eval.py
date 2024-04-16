import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import hhcl
from configs.defaults import get_cfg_defaults
from dataset.dataset import RunnerDataset
from hhcl import datasets
from hhcl.utils.data import transforms as T
from models.gruae import GRUAutoEncoder
from train_hhcl import get_test_loader as get_shoes_loader


def get_cfg(config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.freeze()
    return cfg


def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_hist(cv_img, bins=8, size=None):
    bgr_hist = []
    if size is not None:
        cv_img = cv2.resize(cv_img, size)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        hist = cv2.calcHist([cv_img], [i], None, [bins], [0, 256])
        bgr_hist.append(hist.reshape(bins,))
    bgr_hist = np.array(bgr_hist)
    return bgr_hist.reshape(-1)


def get_upper_hist(cv_img, bins=8, size=None):
    bgr_hist = []
    if size is not None:
        cv_img = cv2.resize(cv_img, size)
    data = cv_img[:cv_img.shape[0]//2, :, :]
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8)
    for i in range(3):
        hist = cv2.calcHist([data], [i], mask, [bins], [0, 1])
        bgr_hist.append(hist.reshape(bins,))
    bgr_hist = np.array(bgr_hist)
    return bgr_hist.reshape(-1)


def get_all_hist(images, shoes_file):
    hist = []
    for img in images:
        img = img.permute(1, 2, 0)
        img = np.array(img)
        hist.append(get_upper_hist(img))
    hist = np.array(hist)
    hist = np.mean(hist, axis=0)

    shoe_img = cv2.imread(shoes_file)
    shoe_hist = get_hist(shoe_img, size=(200, 100))
    return hist, shoe_hist


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def unified_seqlen(batch):
    images = []
    for img, _ in batch:
        images.append(img)

    images_flipped = [img.flip(0) for img in images]
    pad_len = 47 - images_flipped[0].shape[0]
    if pad_len > 0:
        pad_tensor = torch.zeros(pad_len, 3, 64, 64)
        images_flipped[0] = torch.cat([images_flipped[0], pad_tensor], dim=0)

    inputs = rnn.pad_sequence(images_flipped, batch_first=True, padding_value=0)
    inputs = inputs.flip(1)
    del images, images_flipped
    return inputs


def collate_wrapper():
    def collate(batch):
        return unified_seqlen(batch)
    return collate


def get_testloader(dataset):
    collate_fn = collate_wrapper()
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    return test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/gruae/daytime.yaml")
    parser.add_argument("--type", type=str, default="daytime")
    args = parser.parse_args()

    cfg = get_cfg(args.config)
    set_seed(cfg.SEED)

    root_dir = [f"./data/evaluation/{args.type}/images"]
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    gruae_dataset = RunnerDataset(root_dir, transform=transform)
    test_loader = get_testloader(gruae_dataset)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    hhcl_dataset = RunnerDataset(root_dir, transform=test_transformer)
    shoes_dataset = get_data("custom", f"./data/HHCL/shoes/{args.type}")

    # gruae = GRUAutoEncoder(cfg)
    # gruae.load_state_dict(torch.load(f"./output/gruae/{args.type}/checkpoint/last_dict.pth"))
    # gruae.to(cfg.MODEL.DEVICE)
    # gruae.eval()

    # gru_features = []  # (Query, Dim)
    # for x in test_loader:
    #     x = x.to(cfg.MODEL.DEVICE)
    #     with torch.no_grad():
    #         _, hidden = gruae(x, 0)
    #         feature = hidden.squeeze(0).cpu().numpy()
    #         if len(gru_features) == 0:
    #             gru_features.append(feature)
    #             gru_features = np.array(gru_features).squeeze()
    #         else:
    #             gru_features = np.concatenate([gru_features, feature], axis=0)

    # del gruae
    # hhclp = torch.load(f"./output/hhcl/persons/{args.type}/final_model.pth")
    # hhclp.to(cfg.MODEL.DEVICE)
    # hhclp.eval()

    # hhclp_features = []
    # for i in range(len(hhcl_dataset)):
    #     x, _ = hhcl_dataset[i]
    #     x = x.to(cfg.MODEL.DEVICE)
    #     with torch.no_grad():
    #         feature = hhclp(x).cpu().numpy()
    #         feature = np.mean(feature, axis=0)
    #         hhclp_features.append(feature)
    # hhclp_features = np.array(hhclp_features)

    # del hhclp, hhcl_dataset
    # hhcls = torch.load(f"./output/hhcl/shoes/{args.type}/final_model.pth")
    # hhcls.to(cfg.MODEL.DEVICE)
    # hhcls.eval()

    # hhcls_features = []
    # for i in range(len(shoes_dataset.query)):
    #     shoe_img_path, _, _ = shoes_dataset.query[i]
    #     shoe_img = Image.open(shoe_img_path)
    #     x = test_transformer(shoe_img).unsqueeze(0).to(cfg.MODEL.DEVICE)
    #     with torch.no_grad():
    #         feature = hhcls(x).squeeze(0).cpu().numpy()
    #         hhcls_features.append(feature)
    # hhcls_features = np.array(hhcls_features)

    # del hhcls

    person_hist = []
    shoes_hist = []
    for i in range(len(gruae_dataset)):
        hist, shoe_hist = get_all_hist(gruae_dataset[i][0], shoes_dataset.query[i][0])
        person_hist.append(hist)
        shoes_hist.append(shoe_hist)
    hist_features = np.array(person_hist)
    shoe_hist_features = np.array(shoes_hist)

    

if __name__ == "__main__":
    main()
