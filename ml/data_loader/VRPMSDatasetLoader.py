import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import torch

from torchvision.transforms import v2
from torch.utils.data import Dataset


def frequency_ratio():
    """Draw one global scale r ∈ [0.3,0.8]∪[1.2,1.7] and repeat it per frame."""
    r1 = np.random.uniform(0.3, 0.8)
    r2 = np.random.uniform(1.2, 1.7)
    return np.random.choice([r1, r2])


class SixWayAug:
    def __init__(self):
        self.ops = [
            v2.Lambda(lambda x: x),               # identity
            v2.RandomRotation([90, 90]),
            v2.RandomRotation([180, 180]),
            v2.RandomRotation([270, 270]),
            v2.RandomHorizontalFlip(1.0),
            v2.RandomVerticalFlip(1.0),
        ]

    def __call__(self, x):
        return random.choice(self.ops)(x)


class VRPMSDatasetLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The data_loader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, cv_split_files, input_signals, label_signals, config_data, no_labels=False):
        """Inits data_loader with lists of files.

        Args:
            dataset_name(str): name of the data_loader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs_list = list()
        self.labels_list = list()
        self.dataset_name = dataset_name
        self.cached_path = config_data.CACHED_PATH
        self.data_format = config_data.DATA_FORMAT
        self.cv_split_files = cv_split_files
        self.input_signals = input_signals
        self.label_signals = label_signals
        self.input_signal = self.input_signals[0]
        self.label_signal = self.label_signals[0]
        self.no_labels = no_labels
        self.preprocessed_data_len = 0
        self.h = config_data.PREPROCESS.RESIZE.H
        self.w = config_data.PREPROCESS.RESIZE.W

        # Load preprocessed data
        if not os.path.exists(self.cached_path):
            print('Data path:', self.cached_path)
            raise ValueError(self.dataset_name, 'Please preprocess data first. Preprocessed directory does not exist!')
        if len(self.cv_split_files) == 0:
            raise ValueError(self.dataset_name, 'Please provide cv_split_files!')

        print(f'{self.dataset_name}: Data path: {self.cached_path}')
        print(f'{self.dataset_name}: N files: {self.preprocessed_data_len}\n')

        self.by_video = {}                     # video_id -> list[path]
        for clip_path in cv_split_files:
            participant = clip_path[0].split('/')[-1].split('_')[0]
            self.by_video.setdefault(participant, []).extend(clip_path)

        self.video_ids = list(self.by_video.keys())  # used for sampling

        # -------- transforms ------------------------------------------
        self.to_float   = v2.ToDtype(torch.float32, scale=True)   # 0‑255 → 0‑1
        self.aug6       = SixWayAug()
        self.normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.video_ids)


    def _load_clip(self, path):
        """npy → (C,T,H,W) float32 in [0,1] resized"""
        clip = np.load(path).astype(np.float32)          # (C,T,H,W) or (T,C,H,W)
        if clip.shape[0] != 3:                           # if (T,C,...) bring to (C,T,...)
            clip = clip.transpose(1, 0, 2, 3)
        clip = self.to_float(torch.from_numpy(clip)) / 255    # to [0,1]
        # clip = self.to_float(torch.from_numpy(clip))          # to [0,1]
        clip = self.normalize(clip.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        return clip


    def _sample_four_clips(self, clip_list):
        """
        From all clips that belong to ONE video pick four consecutive clips
        that together cover 600 frames (paper). Assumes naming with ...chunk0
        ...chunk1 ...chunk2 ...chunk3.
        """
        # choose start so that we still have 4 successive chunks
        base = random.randrange(0, len(clip_list) // 4) * 4
        return [clip_list[base + i] for i in range(4)]


    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T).
        Transformation is the same for one clip."""

        participant = self.video_ids[index]
        clips_paths = self._sample_four_clips(self.by_video[participant])
        clips = [self._load_clip(p) for p in clips_paths]

        # pick anchor / neighbour
        anchor = clips[0]

        # two positive augmentations of the SAME anchor
        pos1 = self.aug6(anchor.clone())
        pos2 = self.aug6(anchor.clone())
        # pos1 = anchor.clone()
        # pos2 = anchor.clone()

        num_negative = 4
        ratio_array = []
        for i in range(num_negative):
            ratio = np.repeat(frequency_ratio(), anchor.shape[1])
            ratio_array.append(ratio)
        ratio_array = torch.tensor(np.array(ratio_array), dtype=torch.float32)

        # --------------------------------------------------------------
        # labels : here we simply load one label vector per clip‑pair
        # assume each clip has a matching label .npy with shape (T,)
        lab = np.load(clips_paths[0].replace(f'input_{self.input_signal}', f'label_{self.label_signal}'))
        lab_out = torch.from_numpy(lab).float()

        item_path = clips_paths[0]
        item_path_filename = item_path.split(os.sep)[-1]
        filename = item_path_filename.split(f'_input_{self.input_signals[0]}')[0]
        chunk_id = item_path_filename.split(f'_input_{self.input_signals[0]}')[1].split('.')[0]

        return [anchor, pos1, pos2, clips[1], clips[2], clips[3], ratio_array], [lab_out], filename, chunk_id   # keep output signature