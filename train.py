import argparse
import copy
import os
import random
from logging import getLogger, StreamHandler, DEBUG, Formatter

import neptune.new as neptune
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR

from configs.defaults import get_cfg_defaults
from dataset.dataset import get_dataloader
from models.gruae import GRUAutoEncoder

handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger = getLogger("Log")
logger.setLevel(DEBUG)
for h in logger.handlers[:]:
    logger.removeHandler(h)
    h.close()
logger.addHandler(handler)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_neptune(cfg):
    with open(cfg.TOKEN_PATH, "r") as f:
        api_token = f.readline().rstrip("\n")
    run = neptune.init_run(
        project=cfg.PROJECT,
        api_token=api_token,
    )
    return run


def get_cfg(config, opts):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def upload_cfg(cfg, run=None):
    cfg_path = cfg.OUTPUT_DIR + "/config.yaml"
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump())
    if run is not None:
        run["config"].upload(cfg_path)


def create_workspace(cfg):
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    work_dirs = ["/images", "/checkpoint"]
    for work_dir in work_dirs:
        path = cfg.OUTPUT_DIR + work_dir
        if not os.path.exists(path):
            os.makedirs(path)


def criterion(x, y, device):
    mask = (x[:, :, 0, :, :] == 0).all(dim=2)
    mask = (mask == True).all(dim=2)
    masks = []
    for i in range(len(mask)):
        tmp = []
        for j in range(len(mask[i])):
            if mask[i][j] == False:
                tmp.append(torch.ones(3, x.size(3), x.size(4)))
            else:
                tmp.append(torch.zeros(3, x.size(3), x.size(4)))
        tmp = torch.stack(tmp)
        masks.append(tmp)
    masks = torch.stack(masks).to(device)

    diff = (torch.flatten(x) - torch.flatten(y)) ** 2.0 * torch.flatten(masks)
    loss = torch.sum(diff) / torch.sum(masks)
    return loss


def one_epoch(model, dataloader, optimizer, scheduler, device, mode="train", run=None):
    epoch_loss = 0

    for x in dataloader:
        x_decoder = torch.flip(x, [1])
        x = x.to(device)
        x_decoder = x_decoder.to(device)

        y = model(x, x_decoder)
        x = x_decoder[:, 1:, :, :, :]
        y = y[:, :-1, :, :, :]
        del x_decoder

        loss = criterion(x, y, device)
        epoch_loss += loss.item() / len(dataloader)

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if run is not None:
            run[f"{mode}/batch/loss"].append(loss)

    if mode == "train":
        del x, y
        return epoch_loss
    else:
        return epoch_loss, x, y


def save_images(x, y, epoch, output_dir):
    x = x[:, :, :, :, :].permute(0, 1, 3, 4, 2).detach().numpy()
    y = y[:, :, :, :, :].permute(0, 1, 3, 4, 2).detach().numpy()

    for i, (img, rec) in enumerate(zip(x, y)):
        mask = ~(img == 0).all(axis=2).all(axis=1).all(axis=1)
        img = img[mask]
        rec = rec[mask]

        img_concat = np.concatenate(img, axis=1)
        rec_concat = np.concatenate(rec, axis=1)
        res = np.concatenate([img_concat, rec_concat], axis=0)

        im = Image.fromarray((res * 255).astype(np.uint8))
        save_file = os.path.join(output_dir, f"{epoch}_{i}.png")
        im.save(save_file)


def save_checkpoint(model, path):
    model = model.to("cpu")
    torch.save(model, path)
    dict_path = path.replace(".pth", "_dict.pth")
    torch.save(model.state_dict(), dict_path)


def train(cfg, model, train_loader, valid_loader, run=None):
    output_dir = cfg.OUTPUT_DIR
    best_path = output_dir + "/checkpoint/best.pth"
    last_path = output_dir + "/checkpoint/last.pth"

    lr = cfg.TRAIN.LR
    epochs = cfg.TRAIN.EPOCHS
    device = cfg.MODEL.DEVICE

    if cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif cfg.TRAIN.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)

    min_valid_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = one_epoch(model, train_loader, optimizer, scheduler,
                                     device, mode="train", run=run)
        model.eval()
        valid_epoch_loss, x, y = one_epoch(model, valid_loader, optimizer, scheduler,
                                           device, mode="valid", run=run)

        if run is not None:
            run["train/epoch/loss"].append(train_epoch_loss)
            run["valid/epoch/loss"].append(valid_epoch_loss)
        logger.debug("Epoch {} was finished.".format(epoch))
        logger.debug("Training loss: {0:.4f}".format(train_epoch_loss))
        logger.debug("Validation loss: {0:.4f}".format(valid_epoch_loss))

        if valid_epoch_loss < min_valid_loss:
            min_valid_loss = valid_epoch_loss
            best_model = copy.deepcopy(model)
            save_checkpoint(best_model, best_path)
            del best_model

        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            save_images(x.cpu(), y.cpu(), epoch, os.path.join(output_dir, "images"))
        del x, y

    save_checkpoint(model, last_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/gruae/night.yaml")
    parser.add_argument("--output_dir", type=str, default="./output/gruae")
    parser.add_argument("--project", type=str, default="username/project")
    parser.add_argument("--token_path", type=str, default="./token/neptune.txt")
    args = parser.parse_args()

    opts = ["OUTPUT_DIR", args.output_dir, "PROJECT", args.project, "TOKEN_PATH", args.token_path]
    cfg = get_cfg(args.config, opts)
    create_workspace(cfg)
    set_seed(cfg.SEED)
    logger.debug("Config data\n{}\n".format(cfg))

    if cfg.NEPTUNE:
        run = init_neptune(cfg)
    else:
        run = None
    upload_cfg(cfg, run)

    train_loader, valid_loader = get_dataloader(cfg)
    logger.debug("Trainloader length: {}".format(len(train_loader)))
    logger.debug("Validloader length: {}".format(len(valid_loader)))

    model = GRUAutoEncoder(cfg)
    model.to(cfg.MODEL.DEVICE)

    train(cfg, model, train_loader, valid_loader, run)
    run.stop()


if __name__ == "__main__":
    main()
