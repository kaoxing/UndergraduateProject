# author:kaoxing
# date: 2024/4/19

import os
import torch
from data.base_dataset import BaseDataset, get_transform
import SimpleITK as sitk
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def letterbox_image(image, size):
    """
    保持比例的图像缩放用：代替了传统的resize方式
    对图片加以灰色背景，补全较目标比例相差的边缘部分。
    该函数只能处理单张图片
    :param image: 原始图片
    :param size: 目标图像宽高的元组
    :return: 缩放到size后的目标图像
    """
    ret = []
    for img in image:
        minValue = min(size)
        img = Image.fromarray(img)
        iw, ih = img.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        # 此时为按比例缩放，为torch提供的函数
        img = img.resize((nw, nh), Image.BICUBIC)
        # 构建新的RGB背景图片
        new_image = Image.new(img.mode, size, minValue)
        # 缩放后的图片粘贴至背景图片上
        '''参数可选4元组及2元组，如果选择2元组，则为新图片相当于背景图片的左上角坐标'''
        new_image.paste(img, ((w - nw) // 2, (h - nh) // 2))
        ret.append(np.asarray(new_image))
    return np.asarray(ret)


def reverse_letterbox_image(image, raw_size):
    """
    逆letterbox_image，将图像还原为原始大小
    :param raw_size: 原始大小
    :param image: ndarray
    :return:
    """
    ret = []
    for img in image:
        img = Image.fromarray(img)
        # 计算原始图像缩放后的大小
        iw, ih = raw_size
        w, h = img.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        # 裁剪图像
        img = img.crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
        # 将图像还原为原始大小
        img = img.resize(raw_size, Image.BICUBIC)
        ret.append(np.asarray(img))
    return np.asarray(ret)


class niiGzTestDataset(BaseDataset):
    # 读取nnUNet处理后的数据

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        print("niiGzDataset")
        self.paths = opt.dataroot
        self.mean = opt.mean
        self.std = opt.std
        # 获取testA,testB的文件.niigz
        self.dir_A = str(os.path.join(opt.dataroot, opt.phase + 'A'))  # create a path '/path/to/data/trainA'
        self.dir_B = str(os.path.join(opt.dataroot, opt.phase + 'B'))  # create a path '/path/to/data/trainB'
        self.paths_tra = [os.path.join(self.dir_A, i) for i in os.listdir(self.dir_A)]
        self.paths_trb = [os.path.join(self.dir_B, i) for i in os.listdir(self.dir_B)]

        self.transform = [transforms.ToTensor()]
        if opt.norm:
            self.transform.append(transforms.Normalize((self.mean,), (self.std,)))
        self.transform.append(transforms.Lambda(lambda x: letterbox_image(x, (opt.load_size, opt.load_size))))
        self.transform = transforms.Compose(self.transform)

    def __getitem__(self, index):
        # 读取.niigz并transform，一次返回一整个3D图像以及info
        img_a = sitk.ReadImage(self.paths_tra[index])
        img_b = sitk.ReadImage(self.paths_trb[index])
        arr_a = sitk.GetArrayFromImage(img_a)
        arr_b = sitk.GetArrayFromImage(img_b)
        # 数据transform
        arr_a = self.transform(arr_a)
        arr_b = self.transform(arr_b)
        return {
            'img_A': img_a,
            'img_B': img_b,
            'arr_A': arr_a,
            'arr_B': arr_b,
            'path_A': self.paths_tra[index],
            'path_B': self.paths_trb[index]
        }

    def __len__(self):
        return min(len(self.tra), len(self.trb))


if __name__ == '__main__':
    img = np.zeros((100, 100), dtype=np.int16)
    img[:, :] = 255
    img[10:90, 10:90] = 500
    plt.imshow(letterbox_image(img, (200, 150)))
    plt.show()
    plt.imshow(reverse_letterbox_image(letterbox_image(img, (200, 150)), (100, 100)))
    plt.show()
