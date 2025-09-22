import argparse
import json
import numpy as np
import torch

from source.ml.config import get_config
from source.ml.models.PhysNetNew import PhysNetNew
from source.ml.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


def main():
    # %% Data input & output
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='Name of the configuration file')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # %% Load configs
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    device = torch.device('cuda:0')
    model = PhysNetNew(frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH).to(device)


if __name__ == "__main__":
    main()
