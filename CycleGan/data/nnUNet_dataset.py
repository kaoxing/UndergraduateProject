# author kaoxing
# date 2024-4-17

import os

import numpy as np
import torch
from data.base_dataset import BaseDataset, get_transform
import SimpleITK as sitk
from torchvision import transforms


class nnUNetDataset(BaseDataset):
    # 读取nnUNet处理后的数据
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--flip', action="store_true", help='do flip')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        print("nnUNet_dataset")
        self.paths = opt.dataroot
        # 获取trainA trainB的文件.npz
        self.paths_tra = os.listdir(os.path.join(self.paths, 'trainA'))
        self.paths_tra = [os.path.join(self.paths, 'trainA', i) for i in self.paths_tra]
        self.paths_trb = os.listdir(os.path.join(self.paths, 'trainB'))
        self.paths_trb = [os.path.join(self.paths, 'trainB', i) for i in self.paths_trb]

        self.transform = [transforms.ToTensor()]
        if opt.load_size != 0:
            self.transform.append(transforms.Resize(opt.load_size))
        if opt.crop_size != 0:
            self.transform.append(transforms.RandomCrop(opt.crop_size))
        if opt.flip:
            self.transform.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(self.transform)
        # 读取数据,将数据读取为数组,为了方便按index读取，将多个三维数组按通道拼接为一个三维数组
        self.tra = []
        for path in self.paths_tra:
            data = np.load(path)['data']
            self.tra.extend(data)
        self.trb = []
        for path in self.paths_trb:
            data = np.load(path)['data']
            self.trb.extend(data)

    def __getitem__(self, index):
        # 读取数据
        return {
            'data_A': self.transform(self.tra[index]),
            'data_B': self.transform(self.trb[index]),
            'path': ""  # 不需要实现，因为其仅在test时用到
        }

    def __len__(self):
        return min(len(self.tra), len(self.trb))
