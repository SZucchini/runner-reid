# Runner re-identification from single-view running video in the open-world setting
This repository contains the official implementation of "[Runner re-identification from single-view running video in the open-world setting.](https://link.springer.com/article/10.1007/s11042-024-18881-x)" The code supports the methods and experiments presented in the paper. [[arXiv]](https://arxiv.org/abs/2310.11700)

## Overview
This repository contains the following:
- Training code for GRUAE
- Evaluation code with runner dataset
- Trained weights for GRUAE and HHCL with runner dataset
- Feature vectors for evaluation
- Limited image dataset for evaluation
- Customized [HHCL](https://github.com/bupt-ai-cz/HHCL-ReID) scripts for our dataset

This repository does not contain the following:
- Full dataset used in the paper (due to privacy issues)
- Original evaluation dataset (anonymized due to privacy issues)

From above reasons, you can not reproduce the results in the paper from scratch. However, you can reproduce the results in the paper using evaluation script and pre-calculated features.

## Getting Started
### Installation
1. Clone this repository: `$ git clone https://github.com/SZucchini/runner-reid.git`
2. Create conda environment: `$ conda env create --file env.frozen.yaml`

### Evaluation
Example of evaluation:
```
python ./eval.py --type daytime \
    --config ./configs/gruae/daytime_eval.yaml \
    --use_embedding
```

### Optional
You can download trained weights [[here]](https://drive.google.com/drive/folders/11M49cKsJ2jWcpfYu1YwHbNmeVNiBok0r?usp=sharing) and evaluation images example [[here]](https://drive.google.com/drive/folders/1YZgXD8Ey1NVaiifksGaiBPwXlYNePD0n?usp=sharing) from Google Drive.

## Citation
If you find this repository useful, please cite our paper:
```
@article{suzuki2024runner,
  title={Runner re-identification from single-view running video in the open-world setting},
  author={Suzuki, Tomohiro and Tsutsui, Kazushi and Takeda, Kazuya and Fujii, Keisuke},
  journal={Multimedia Tools and Applications},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```

## Contact
If you have any questions, please contact author:
- Tomohiro Suzuki (suzuki.tomohiro[at]g.sp.m.is.nagoya-u.ac.jp)
