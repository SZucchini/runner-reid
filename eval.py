import argparse
import os
import random

import cv2
import numpy as np
import torch
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


def get_gt(gt_file):
    gt = {}
    with open(gt_file) as f:
        for s_line in f:
            s_line = s_line.replace("\n", "")
            key, value = s_line.split(',')
            gt[key] = [int(value), 0, 0]
    return gt


def get_annotation(annotation_file):
    annotation = {}
    start_frames = []
    end_frames = []

    cnt = 0
    with open(annotation_file) as f:
        for s_line in f:
            s_line = s_line.replace("\n", "")
            frame_start = int(s_line.split(',')[1])
            frame_end = int(s_line.split(',')[2])

            value = s_line.split(',')[-1]
            annotation[str(cnt)] = value
            start_frames.append(int(frame_start))
            end_frames.append(int(frame_end))
            cnt += 1

    return annotation, start_frames, end_frames


def calc_hist_sim(hist1, hist2):
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return corr


def cos_sim(v1, v2):
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim


def calc_mat(features, candidates):
    mat = np.zeros((len(features), len(features)))
    for i in range(len(features)):
        for j in range(len(features)):
            sim = cos_sim(features[i], features[j])
            if j not in candidates[i]:
                sim = 0
            mat[i, j] = sim
    return mat


def calc_mAP(mat, gt, annotation_data):
    ave_mAP = 0
    person_mAP = {}
    num_persons = 0
    for pid in range(len(mat)):
        query = annotation_data[str(pid)]
        if 'sprinter' in query or 'rugby' in query:
            continue
        num_persons += 1
        if query not in person_mAP.keys():
            person_mAP[query] = [0, 0]
        num_target = gt[query][0] - 1

        ans = np.argsort(mat[pid])[::-1]
        mAP = 0
        cnt = 0
        for i, a in enumerate(ans):
            p = annotation_data[str(a)]
            if p == query:
                cnt += 1
                mAP += cnt / (i+1)
            if cnt == num_target:
                break

        mAP /= num_target
        person_mAP[query][0] += mAP
        person_mAP[query][1] += 1
        ave_mAP += mAP

    ave_mAP /= num_persons
    return ave_mAP, person_mAP


def rankn_acc(mat, n, annotation_data):
    cnt = 0
    skip_cnt = 0
    for pid in range(len(mat)):
        query = annotation_data[str(pid)]
        if 'sprinter' in query:
            skip_cnt += 1
            continue

        ans = np.argsort(mat[pid])[::-1][:n]
        res = []
        for a in ans:
            res.append(annotation_data[str(a)])

        if query in res:
            cnt += 1

    return cnt / (len(mat) - skip_cnt)


def cmc_curve(mat, n, annotation_data):
    res = []
    for i in range(1, n+1):
        res.append(rankn_acc(mat, i, annotation_data))
    return res


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
    parser.add_argument("--use_embedding", action="store_true")
    args = parser.parse_args()

    cfg = get_cfg(args.config)
    set_seed(cfg.SEED)

    if not args.use_embedding:
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

        gruae = GRUAutoEncoder(cfg)
        gruae.load_state_dict(torch.load(f"./output/gruae/{args.type}/checkpoint/last_dict.pth"))
        gruae.to(cfg.MODEL.DEVICE)
        gruae.eval()

        gru_features = []  # (Query, Dim)
        for x in test_loader:
            x = x.to(cfg.MODEL.DEVICE)
            with torch.no_grad():
                _, hidden = gruae(x, 0)
                feature = hidden.squeeze(0).cpu().numpy()
                if len(gru_features) == 0:
                    gru_features.append(feature)
                    gru_features = np.array(gru_features).squeeze()
                else:
                    gru_features = np.concatenate([gru_features, feature], axis=0)

        del gruae
        hhclp = torch.load(f"./output/hhcl/persons/{args.type}/checkpoint/final_model_rep.pth")
        hhclp.to(cfg.MODEL.DEVICE)
        hhclp.eval()

        hhclp_features = []
        for i in range(len(hhcl_dataset)):
            x, _ = hhcl_dataset[i]
            x = x.to(cfg.MODEL.DEVICE)
            with torch.no_grad():
                feature = hhclp(x).cpu().numpy()
                feature = np.mean(feature, axis=0)
                hhclp_features.append(feature)
        hhclp_features = np.array(hhclp_features)

        del hhclp, hhcl_dataset
        hhcls = torch.load(f"./output/hhcl/shoes/{args.type}/checkpoint/final_model_rep.pth")
        hhcls.to(cfg.MODEL.DEVICE)
        hhcls.eval()

        hhcls_features = []
        for i in range(len(shoes_dataset.query)):
            shoe_img_path, _, _ = shoes_dataset.query[i]
            shoe_img = Image.open(shoe_img_path)
            x = test_transformer(shoe_img).unsqueeze(0).to(cfg.MODEL.DEVICE)
            with torch.no_grad():
                feature = hhcls(x).squeeze(0).cpu().numpy()
                hhcls_features.append(feature)
        hhcls_features = np.array(hhcls_features)

        del hhcls

        person_hist = []
        shoes_hist = []
        for i in range(len(gruae_dataset)):
            hist, shoe_hist = get_all_hist(gruae_dataset[i][0], shoes_dataset.query[i][0])
            person_hist.append(hist)
            shoes_hist.append(shoe_hist)
        hist_features = np.array(person_hist)
        shoe_hist_features = np.array(shoes_hist)

    else:
        gru_features = np.load(f"./data/evaluation/{args.type}/features/gru_features.npy")
        hhclp_features = np.load(f"./data/evaluation/{args.type}/features/hhclp_features.npy")
        hhcls_features = np.load(f"./data/evaluation/{args.type}/features/hhcls_features.npy")
        hist_features = np.load(f"./data/evaluation/{args.type}/features/hist_features.npy")
        shoe_hist_features = np.load(
            f"./data/evaluation/{args.type}/features/shoe_hist_features.npy"
        )

    gt = get_gt(f"./data/evaluation/{args.type}/gt_full.txt")
    annotation_file = f"./data/evaluation/{args.type}/scene.txt"
    annotation_data, start_frames, end_frames = get_annotation(annotation_file)

    start_frames = np.array(start_frames)
    end_frames = np.array(end_frames)
    candidates = []
    for i in range(len(start_frames)):
        candidates.append(np.where((start_frames - start_frames[i] > 3600) |
                                   (start_frames - start_frames[i] < -3600))[0].tolist())

    mat_histwoshoe = np.zeros((len(hist_features), len(hist_features)))
    for i in range(len(hist_features)):
        for j in range(len(hist_features)):
            img_sim = calc_hist_sim(hist_features[i], hist_features[j])
            sim = img_sim
            if j not in candidates[i]:
                sim = 0
            mat_histwoshoe[i, j] = sim

    mat_hist = np.zeros((len(hist_features), len(hist_features)))
    for i in range(len(hist_features)):
        for j in range(len(hist_features)):
            img_sim = calc_hist_sim(hist_features[i], hist_features[j])
            shoe_sim = calc_hist_sim(shoe_hist_features[i], shoe_hist_features[j])
            sim = img_sim * 0.9 + shoe_sim * 0.1
            if j not in candidates[i]:
                sim = 0
            mat_hist[i, j] = sim

    mat_gru = calc_mat(gru_features, candidates)
    mat_hhcl = calc_mat(hhclp_features, candidates)

    mat_gru_hist = np.zeros((len(gru_features), len(gru_features)))
    for i in range(len(gru_features)):
        for j in range(len(gru_features)):
            sim = mat_gru[i, j] * 0.85 + mat_hist[i, j] * 0.15
            if j not in candidates[i]:
                sim = 0
            mat_gru_hist[i, j] = sim

    mat_hhcl_shoe = np.zeros((len(hhclp_features), len(hhclp_features)))
    for i in range(len(hhclp_features)):
        for j in range(len(hhclp_features)):
            sim = (cos_sim(hhclp_features[i], hhclp_features[j]) * 0.75
                   + cos_sim(hhcls_features[i], hhcls_features[j]) * 0.25)
            if j not in candidates[i]:
                sim = 0
            mat_hhcl_shoe[i, j] = sim

    mAP_histwoshoe, pmAP_histwoshoe = calc_mAP(mat_histwoshoe, gt, annotation_data)
    mAP_hist, pmAP_hist = calc_mAP(mat_hist, gt, annotation_data)
    mAP_gru, pmAP_gru = calc_mAP(mat_gru, gt, annotation_data)
    mAP_gru_hist, pmAP_gru_hist = calc_mAP(mat_gru_hist, gt, annotation_data)
    mAP_hhcl, pmAP_hhcl = calc_mAP(mat_hhcl, gt, annotation_data)
    mAP_hhcl_shoe, pmAP_hhcl_shoe = calc_mAP(mat_hhcl_shoe, gt, annotation_data)

    print("mAP Hist w/o Shoe:", mAP_histwoshoe)
    print("mAP Hist:", mAP_hist)
    print("mAP GRU:", mAP_gru)
    print("mAP GRU Hist:", mAP_gru_hist)
    print("mAP HHCL:", mAP_hhcl)
    print("mAP HHCL Shoe:", mAP_hhcl_shoe)

    rank = 10
    histwoshoe_cmc = cmc_curve(mat_histwoshoe, rank, annotation_data)
    hist_cmc = cmc_curve(mat_hist, rank, annotation_data)
    gru_cmc = cmc_curve(mat_gru, rank, annotation_data)
    gru_hist_cmc = cmc_curve(mat_gru_hist, rank, annotation_data)
    hhcl_cmc = cmc_curve(mat_hhcl, rank, annotation_data)
    hhcl_shoe_cmc = cmc_curve(mat_hhcl_shoe, rank, annotation_data)

    for i in range(rank):
        print(f"Rank {i+1} Acc:")
        print(f"Hist w/o Shoe: {histwoshoe_cmc[i]}")
        print(f"Hist: {hist_cmc[i]}")
        print(f"GRU: {gru_cmc[i]}")
        print(f"GRU Hist: {gru_hist_cmc[i]}")
        print(f"HHCL: {hhcl_cmc[i]}")
        print(f"HHCL Shoe: {hhcl_shoe_cmc[i]}")


if __name__ == "__main__":
    main()
