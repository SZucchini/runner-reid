from yacs.config import CfgNode as CN


_C = CN()

_C.DATASETS = CN()
_C.DATASETS.DAYTIME_ROOT = "./data/GRUAE/large/daytime"
_C.DATASETS.NIGHT_ROOT = "./data/GRUAE/large/night"
_C.DATASETS.TYPE = "DAYTIME"  # or NIGHT

_C.INPUT = CN()
_C.INPUT.BATCH_SIZE = 32
_C.INPUT.MAX_SEQ_LEN = 47
_C.INPUT.IMAGE_SIZE = [64, 64]

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.EVAL = False
_C.MODEL.LATENT_DIM = 128

_C.NEPTUNE = CN()
_C.NEPTUNE.PROJECT = "username/project"
_C.NEPTUNE.TOKEN_PATH = "./token/neptune.txt"
_C.NEPTUNE.USE = False

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 500
_C.TRAIN.LR = 0.001
_C.TRAIN.OPTIMIZER = "Adam"

_C.OUTPUT_DIR = "./logs/output"
_C.SEED = 42


def get_cfg_defaults():
    return _C.clone()
