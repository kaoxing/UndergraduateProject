# author kaoxing
# date 2024-4-17

import os

import numpy as np
from data.base_dataset import BaseDataset, get_transform
import SimpleITK as sitk
from torchvision import transforms


class niigzdataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.paths = opt.dataroot
        # 获取trainA trainB testA testB的文件
        self.paths_tra = os.listdir(os.path.join(self.paths, 'trainA'))
        self.paths_tra = [os.path.join(self.paths, 'trainA', i) for i in self.paths_tra]
        self.paths_trb = os.listdir(os.path.join(self.paths, 'trainB'))
        self.paths_trb = [os.path.join(self.paths, 'trainB', i) for i in self.paths_trb]
        # 获取路径,读取数据,将数据读取为数组,为了方便按index读取，将多个三维数组按通道拼接为一个三维数组
        transform = transforms.ToTensor()
        self.tra = []
        for path in self.paths_tra:
            img = sitk.ReadImage(path)
            img = sitk.GetArrayFromImage(img)
            self.tra.extend(img)
        self.tra = transform(self.tra)
        self.trb = []
        for path in self.paths_trb:
            img = sitk.ReadImage(path)
            img = sitk.GetArrayFromImage(img)
            self.trb.extend(img)
        self.trb = transform(self.trb)
        # 定义transform
        self.transform = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        # 读取数据
        return {
            'data_A': self.transform(self.tra[index]),
            'data_B': self.transform(self.trb[index]),
            'path': ""  # 不需要实现，因为其仅在test时用到
        }

    def __len__(self):
        return min(len(self.tra), len(self.trb))
