import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from torchvision.transforms import v2
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
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

        self.load_preprocessed_data()
        print(f'{self.dataset_name}: Data path: {self.cached_path}')
        print(f'{self.dataset_name}: N files: {self.preprocessed_data_len}\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs_list[0])

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T).
        Transformation is the same for one clip."""
        data_out = []
        if self.dataset_name == 'Train':
            transforms = v2.Compose([
                # v2.RandomCrop(size=(self.h // 2, self.w // 2)),
                # v2.Resize((self.h, self.w)),
                # v2.RandomApply([v2.RandomCrop(size=(self.h // 2, self.w)),
                #                 v2.Pad((0, self.h // 4)),], p=0.5),
                # v2.RandomApply([v2.RandomCrop(size=(self.h, self.w // 2)),
                #                 v2.Pad((self.w // 4, 0)), ], p=0.5),
                # v2.RandomApply([v2.RandomCrop(size=(self.h // 2, self.w // 2)),
                #                 v2.Resize((self.h, self.w)), ], p=0.5),
                # v2.RandomCrop(size=(self.h, self.w // 2)),
                # v2.Pad((self.w // 4, 0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomCrop(size=(self.h // 2, self.w)),
                v2.Pad((0, self.h // 4)),
                v2.RandomRotation(degrees=(-20, 20)),

                v2.ToDtype(torch.float32, scale=False),
            ])
        else:
            transforms = v2.Compose([
                v2.ToDtype(torch.float32, scale=False),
            ])
        # print(index)
        for i_input in range(len(self.inputs_list)):
            # try:
            data_temp = torch.from_numpy(np.load(self.inputs_list[i_input][index]))
            # except:
            #     print('Error loading i_input:', i_input)
            #     print('Error loading index:', index)
            #     print('Error loading:', self.inputs_list[i_input][index])
            if self.input_signals[i_input] in ['face', 'hand', 'et']:
                data_temp = transforms(data_temp)
                if self.data_format == 'NDCHW':
                    data_temp = data_temp.permute(1, 0, 2, 3)
            else:
                data_temp = data_temp.type(torch.float32)
            data_out.append(data_temp)

        labels_out = []
        if not self.no_labels:  # NEW
            for i_label in range(len(self.labels_list)):
                labels_out.append(np.float32(np.load(self.labels_list[i_label][index])))

        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs_list[0][index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        # split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        # filename = item_path_filename.split('_')[0]
        filename = item_path_filename.split(f'_input_{self.input_signals[0]}')[0]  # NEW
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        # chunk_id = item_path_filename.split('_')[2].split('.')[0][len(self.input_signals[0]):]
        chunk_id = item_path_filename.split(f'_input_{self.input_signals[0]}')[1].split('.')[0]  # NEW
        return data_out, labels_out, filename, chunk_id

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.
        Args:
            None
        Returns:
            None
        """
        inputs = [r for split_file in self.cv_split_files for r in split_file]
        if len(inputs) == 0:
            raise ValueError(self.dataset_name + ' dataset loading data error!')

        inputs_list = []
        for input_signal in self.input_signals:
            inputs_list.append([input_file.replace(f"input_{self.input_signals[0]}", "input_" + input_signal)
                                for input_file in inputs])

        labels_list = []
        for label_signal in self.label_signals:
            labels_list.append([input_file.replace(f"input_{self.input_signals[0]}", "label_" + label_signal)
                                for input_file in inputs])

        self.inputs_list = inputs_list
        self.labels_list = labels_list
        self.preprocessed_data_len = len(inputs)
