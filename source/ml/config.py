# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------\
_C.TOOLBOX_MODE = ""
_C.INPUT_SIGNALS = []
_C.LABEL_SIGNALS = []
_C.TASKS_TO_USE = []
_C.TASKS_EVALUATE = []
_C.LABEL_VALID = 0
_C.SPLIT_METHOD = ""
_C.K_FOLD_SPLITS = 0
_C.NAME_EXTENSION = ""
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LR = 1e-4
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-4
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.PLOT_LOSSES_AND_LR = True
# Train.Data settings
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.INFO = CN()
_C.TRAIN.DATA.FS = 0
_C.TRAIN.DATA.DATA_PATH = ''
_C.TRAIN.DATA.CACHED_PATH = 'PreprocessedData'
_C.TRAIN.DATA.FILE_LIST_PATH = os.path.join(_C.TRAIN.DATA.CACHED_PATH, 'DataFileLists')
_C.TRAIN.DATA.DATASET = ''
_C.TRAIN.DATA.DATA_FORMAT = ''
_C.TRAIN.DATA.DO_PREPROCESS = False
# Train Data preprocessing
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TRAIN.DATA.PREPROCESS.DOWNSAMPLE = 1
_C.TRAIN.DATA.PREPROCESS.UPSAMPLE = 1
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH_OLD = None
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH = None
_C.TRAIN.DATA.PREPROCESS.DETECTION_LENGTH = None
_C.TRAIN.DATA.PREPROCESS.RESIZE = CN()
_C.TRAIN.DATA.PREPROCESS.RESIZE.W = None
_C.TRAIN.DATA.PREPROCESS.RESIZE.H = None

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------\
_C.TEST = CN()
_C.TEST.OUTPUT_SAVE_DIR = ''
# _C.TEST.METRICS = None
_C.TEST.USE_LAST_EPOCH = True
# Test.Data settings
_C.TEST.DATA = CN()
_C.TEST.DATA.INFO = CN()
_C.TEST.DATA.FS = 0
_C.TEST.DATA.DATA_PATH = ''
_C.TEST.DATA.EXP_DATA_NAME = ''
_C.TEST.DATA.CACHED_PATH = 'PreprocessedData'
_C.TEST.DATA.FILE_LIST_PATH = os.path.join(_C.TEST.DATA.CACHED_PATH, 'DataFileLists')
_C.TEST.DATA.DATASET = ''
_C.TEST.DATA.DATA_FORMAT = ''
_C.TEST.DATA.DO_PREPROCESS = False
# Test Data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH_OLD = None
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH = None
_C.TEST.DATA.PREPROCESS.DOWNSAMPLE = 1
_C.TEST.DATA.PREPROCESS.UPSAMPLE = 1
_C.TEST.DATA.PREPROCESS.DETECTION_LENGTH = None
_C.TEST.DATA.PREPROCESS.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.RESIZE.W = None
_C.TEST.DATA.PREPROCESS.RESIZE.H = None

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = ''
# Dropout rate
_C.MODEL.PATH_MODEL = ''

# -----------------------------------------------------------------------------
# Model Settings for TS-CAN
# -----------------------------------------------------------------------------
_C.MODEL.TSCAN = CN()
_C.MODEL.TSCAN.FRAME_DEPTH = 10

# -----------------------------------------------------------------------------
# Model Settings for PhysFormer
# -----------------------------------------------------------------------------
_C.MODEL.PHYSFORMER = CN()
_C.MODEL.PHYSFORMER.PATCH_SIZE = 4
_C.MODEL.PHYSFORMER.DIM = 96
_C.MODEL.PHYSFORMER.FF_DIM = 144
_C.MODEL.PHYSFORMER.NUM_HEADS = 4
_C.MODEL.PHYSFORMER.NUM_LAYERS = 12
_C.MODEL.PHYSFORMER.THETA = 0.7

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.NO_LABELS = False
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.USE_BEST_EPOCH = False
_C.INFERENCE.EVALUATION_METHOD = 'FFT'
_C.INFERENCE.EVALUATION_WINDOW = CN()
_C.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = False
_C.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = 10
_C.INFERENCE.MODEL_PATH = ''

# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cuda:0"
_C.NUM_OF_GPU_TRAIN = 1

# -----------------------------------------------------------------------------
# Log settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH_TRAINING = None
_C.LOG.PATH_TESTING = None

# -----------------------------------------------------------------------------
# Path settings
# -----------------------------------------------------------------------------
_C.DATA_PATH = None
_C.FILE_PATH = None


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> Merging a config file from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    _update_config_from_file(config, args.cfg_path)
    config.defrost()

    # Set tasks to evaluate if tasks are not None
    config.TASKS_EVALUATE = config.TASKS_TO_USE.copy()
    config.TASKS_EVALUATE.append('overall')

    # Check that validation label in label signals
    if config.LABEL_VALID + 1 > len(config.LABEL_SIGNALS):
        raise ValueError("Validation label not in label signals!")

    # Define preprocessing configuration of training dataset
    config_preprocessing_train = f'CL{config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH_OLD}_' \
                                 f'W{config.TRAIN.DATA.PREPROCESS.RESIZE.W}_' \
                                 f'H{config.TRAIN.DATA.PREPROCESS.RESIZE.H}_' \
                                 f'LabelRaw_VideoTypeRaw'

    # Define preprocessing extended configuration of training dataset
    config_preprocessing_extended_train = f'CL{config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH}_' \
                                          f'Down{config.TRAIN.DATA.PREPROCESS.DOWNSAMPLE}_' \
                                          f'W{config.TRAIN.DATA.PREPROCESS.RESIZE.W}_' \
                                          f'H{config.TRAIN.DATA.PREPROCESS.RESIZE.H}_' \
                                          f'Label{config.TRAIN.DATA.PREPROCESS.LABEL_TYPE}_' \
                                          f'VideoType'
    for data_type in config.TRAIN.DATA.PREPROCESS.DATA_TYPE:
        config_preprocessing_extended_train += data_type
    if config.TRAIN.DATA.PREPROCESS.UPSAMPLE > 1:
        config_preprocessing_extended_train += f'_Up{config.TRAIN.DATA.PREPROCESS.UPSAMPLE}'

    # Define preprocessing configuration of testing dataset
    config_preprocessing_test = f'CL{config.TEST.DATA.PREPROCESS.CHUNK_LENGTH_OLD}_' \
                                f'W{config.TEST.DATA.PREPROCESS.RESIZE.W}_' \
                                f'H{config.TEST.DATA.PREPROCESS.RESIZE.H}_' \
                                f'LabelRaw_VideoTypeRaw'

    # Define preprocessing extended configuration of testing dataset
    config_preprocessing_extended_test = f'CL{config.TEST.DATA.PREPROCESS.CHUNK_LENGTH}_' \
                                         f'Down{config.TEST.DATA.PREPROCESS.DOWNSAMPLE}_' \
                                         f'W{config.TEST.DATA.PREPROCESS.RESIZE.W}_' \
                                         f'H{config.TEST.DATA.PREPROCESS.RESIZE.H}_' \
                                         f'Label{config.TEST.DATA.PREPROCESS.LABEL_TYPE}_' \
                                         f'VideoType'
    for data_type in config.TEST.DATA.PREPROCESS.DATA_TYPE:
        config_preprocessing_extended_test += data_type
    if config.TEST.DATA.PREPROCESS.UPSAMPLE > 1:
        config_preprocessing_extended_test += f'_Up{config.TEST.DATA.PREPROCESS.UPSAMPLE}'

    # Set the data paths
    config.TRAIN.DATA.CACHED_PATH = (config.DATA_PATH + config.TRAIN.DATA.CACHED_PATH + '/' +
                                     config_preprocessing_train + '/' + config_preprocessing_extended_train)
    config.TEST.DATA.CACHED_PATH = (config.DATA_PATH + config.TEST.DATA.CACHED_PATH + '/' +
                                    config_preprocessing_test + '/' + config_preprocessing_extended_test)

    # Set the model and log paths
    config.MODEL.PATH_MODEL = (config.DATA_PATH + config.MODEL.PATH_MODEL +
                               f'/{config.TRAIN.DATA.DATASET}/inputs_{"_".join(config.INPUT_SIGNALS)}'
                               f'/labels_{"_".join(config.LABEL_SIGNALS)}'
                               f'/{config_preprocessing_extended_train}/{config.MODEL.NAME}{config.NAME_EXTENSION}')
    config.LOG.PATH_TRAINING = (config.FILE_PATH + config.LOG.PATH_TRAINING +
                                (f'/{config.TRAIN.DATA.DATASET}/inputs_{"_".join(config.INPUT_SIGNALS)}'
                                 f'/labels_{"_".join(config.LABEL_SIGNALS)}/'
                                 f'{config_preprocessing_extended_train}/{config.MODEL.NAME}{config.NAME_EXTENSION}'))
    config.LOG.PATH_TESTING = (config.FILE_PATH + config.LOG.PATH_TESTING +
                               (f'/{config.TEST.DATA.DATASET}/inputs_{"_".join(config.INPUT_SIGNALS)}'
                                f'/labels_{"_".join(config.LABEL_SIGNALS)}/{config_preprocessing_extended_test}/'
                                f'{config.MODEL.NAME}{config.NAME_EXTENSION}'))

    # Adjust the data sampling frequency to the downsampling
    config.TRAIN.DATA.FS = int(config.TRAIN.DATA.FS / config.TRAIN.DATA.PREPROCESS.DOWNSAMPLE)
    config.TEST.DATA.FS = int(config.TEST.DATA.FS / config.TEST.DATA.PREPROCESS.DOWNSAMPLE)

    # Freeze configs
    config.freeze()

    return config

