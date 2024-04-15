import argparse
import collections
import os
import os.path as osp
import random
import time
from datetime import timedelta
from logging import getLogger, StreamHandler, DEBUG, Formatter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch import nn
from torch.backends import cudnn

from configs.hhcl_defaults import get_cfg_defaults
from models.hhcl.cm import ClusterMemory
from models.hhcl.dataset import RunnerCustom
from models.hhcl.trainer import Trainer
from models.hhcl.utils.dataloader import get_test_loader, get_train_loader
from models.hhcl.utils.extraction import extract_features
from models.hhcl.utils.faiss_rerank import compute_jaccard_distance

start_epoch = 0
best_mAP = 0

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


def get_cfg(config, opts=None):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    if opts is not None:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def upload_cfg(cfg):
    cfg_path = cfg.OUTPUT_DIR + "/config.yaml"
    with open(cfg_path, 'w') as f:
        f.write(cfg.dump())


def create_workspace(cfg):
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    work_dirs = ["/checkpoint"]
    for work_dir in work_dirs:
        path = cfg.OUTPUT_DIR + work_dir
        if not os.path.exists(path):
            os.makedirs(path)


def custom_forward(self, x_encoder, x_decoder=None):
    self.batch_size = x_encoder.size()[0]
    self.seq_length = x_encoder.size()[1]

    x_encoder = x_encoder.view(-1, 3, self.img_size[0], self.img_size[1])
    x_encoder = self.image_encoder(x_encoder)
    x_encoder = x_encoder.reshape(self.batch_size, self.seq_length, -1)

    h = torch.randn(1, self.batch_size, self.latent_dim, requires_grad=True)
    h = h.to(x_encoder.device)

    encoder_output, encoder_hidden = self.encoder_gru(x_encoder, h)
    return encoder_hidden[0]


def load_model(cfg):

    return model


def save_checkpoint(model, path):
    model = model.to("cpu")
    torch.save(model, path)
    dict_path = path.replace(".pth", "_dict.pth")
    torch.save(model.state_dict(), dict_path)


def train(cfg):
    global start_epoch, best_mAP

    start_time = time.monotonic()
    cudnn.benchmark = True

    if cfg.DATASETS.TYPE == "DAYTIME":
        root_dir = cfg.DATASETS.DAYTIME_ROOT
    elif cfg.DATASETS.TYPE == "NIGHT":
        root_dir = cfg.DATASETS.NIGHT_ROOT
    dataset = RunnerCustom(root=root_dir, verbose=True)

    model = load_model(cfg)
    if cfg.TRAIN.MULTI_GPU:
        model = nn.DataParallel(model)
    model.to(cfg.MODEL.DEVICE)
    logger.debug("Model loaded!")

    trainer = Trainer(model)

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.TRAIN.STEP_SIZE, gamma=0.1)

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        with torch.no_grad():
            cluster_loader = get_test_loader(cfg, dataset, testset=dataset.train)

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f[0]].unsqueeze(0) for f, _, _ in dataset.train], 0)
            rerank_dist = compute_jaccard_distance(features, k1=cfg.TRAIN.KONE, k2=cfg.TRAIN.KTWO)

            if epoch == start_epoch and not cfg.TRAIN.DYNAMIC_EPS:
                eps = cfg.TRAIN.EPS
                logger.debug('Clustering criterion eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            else:
                logger.debug('Clustering criterion eps: dynamic')
                eps = cfg.TRAIN.EPS
                while True:
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
                    pseudo_labels = cluster.fit_predict(rerank_dist)
                    cnt_outliers = (pseudo_labels == -1).sum()

                    if cnt_outliers < 0.4 * len(pseudo_labels):
                        logger.debug('DBSCAN clustering finished, eps: {:.3f}'.format(eps))
                        break
                    else:
                        eps += 0.01

                    if eps > 1:
                        logger.debug('DBSCAN clustering failed, eps: {:.3f}'.format(eps))
                        break

            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            cnt = 0
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    cnt += 1
                    continue
                centers[labels[i]].append(features[i])

            logger.debug('Labels length: {}'.format(len(labels)))
            logger.debug('Cluster {} has {} samples'.format(-1, cnt))

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)

        del cluster_loader, features

        memory = ClusterMemory(0, num_cluster, temp=cfg.TRAIN.TEMP,
                               momentum=cfg.TRAIN.MOMENTUM, mode=cfg.TRAIN.MBANK,
                               smooth=cfg.TRAIN.SMOOTH, num_instances=cfg.INPUT.INSTANCES).cuda()
        memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()
        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((imgs_path, _, cid), label) in enumerate(zip(dataset.train, pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((imgs_path, label.item(), cid))
        logger.debug('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(cfg, dataset, trainset=pseudo_labeled_dataset)
        logger.debug("Got train loader!")

        train_loader.new_epoch()
        logger.debug("New epoch!")

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=cfg.PRINT_FREQ, train_iters=len(train_loader))
        logger.debug("Trained!")

        lr_scheduler.step()

    end_time = time.monotonic()
    logger.debug('Total running time: {}'.format(timedelta(seconds=end_time - start_time)))

    if cfg.TRAIN.MULTI_GPU:
        model = model.module
    save_path = osp.join(cfg.OUTPUT_DIR, 'final.pth')
    save_checkpoint(model, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/hhcl/night.yaml")
    parser.add_argument("--output_dir", type=str, default="./logs/output")
    args = parser.parse_args()

    opts = ["OUTPUT_DIR", args.output_dir]
    cfg = get_cfg(args.config, opts)
    logger.debug("Config data\n{}\n".format(cfg))

    set_seed(cfg.SEED)
    upload_cfg(cfg)
    create_workspace(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
