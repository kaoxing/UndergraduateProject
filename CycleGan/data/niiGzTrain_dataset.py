# author:kaoxing
# date: 2024/4/19

from CycleGanDataPreprogress import CycleGANDataPreprocessor
import os
import torch
from data.base_dataset import BaseDataset, get_transform
import SimpleITK as sitk
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class letterbox_image:
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, image: torch.Tensor):
        """
        保持比例的图像缩放用：代替了传统的resize方式
        对图片加以灰色背景，补全较目标比例相差的边缘部分。
        该函数只能处理单张图片
        :param image: 原始图片 Tensor
        :param size: 目标图像宽高的元组
        :return: 缩放到size后的目标图像 Tensor
        """
        # Tensor转为ndarray
        image: np.ndarray = image.cpu().numpy()
        minValue = image.min()
        img = Image.fromarray(image[0])  # tensor自动增加0维，不符合Image格式
        iw, ih = img.size
        # 若原图两轴差距有一倍以上，则将较小的轴放大一倍
        if iw / ih > 2 or ih / iw > 2:
            if iw > ih:
                img = img.resize((iw, ih * 2))  # 若宽度大于高度，则高度放大一倍
            else:
                img = img.resize((iw * 2, ih))  # 若高度大于宽度，则宽度放大一倍
            iw, ih = img.size
        w, h = self.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        # 此时为按比例缩放，为torch提供的函数
        img = img.resize((nw, nh))
        # 构建新的RGB背景图片
        new_image = Image.new(img.mode, self.size, minValue)
        # 缩放后的图片粘贴至背景图片上
        '''参数可选4元组及2元组，如果选择2元组，则为新图片相当于背景图片的左上角坐标'''
        new_image.paste(img, ((w - nw) // 2, (h - nh) // 2))
        img = np.asarray(new_image)
        # ndarray转为Tensor
        img = torch.from_numpy(img).cuda()
        # 还原被去掉的0维
        return img.unsqueeze(0)


class reverse_letterbox_image:

    def __call__(self, image: torch.Tensor, raw_size):
        """
        逆letterbox_image，将图像还原为原始大小
        :param raw_size: 原始大小
        :param image: Tensor
        :return: ndarray
        """
        image = image.cpu().numpy()
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
            img = img.resize(raw_size)
            ret.append(np.asarray(img))
        return np.asarray(ret)


class niiGzTrainDataset(BaseDataset):
    # 读取niigz数据
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
            :param parser:
        """
        parser.add_argument('--flip', action="store_true", help='do flip')
        parser.add_argument('--resize', action="store_true", help='resize')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        print("niiGzDataset")
        self.paths = opt.dataroot

        # 获取testA,testB的文件.niigz
        self.dir_A = str(os.path.join(opt.dataroot, opt.phase + 'A'))  # create a path '/path/to/data/trainA'
        self.dir_B = str(os.path.join(opt.dataroot, opt.phase + 'B'))  # create a path '/path/to/data/trainB'
        self.tra = [os.path.join(self.dir_A, i) for i in os.listdir(self.dir_A)]
        self.trb = [os.path.join(self.dir_B, i) for i in os.listdir(self.dir_B)]
        # 将.nii.gz和.npy文件分离
        self.paths_tra = [i for i in self.tra if i.endswith('.nii.gz')]
        self.paths_trb = [i for i in self.trb if i.endswith('.nii.gz')]


        # 读取数据,将数据读取为数组,为了方便按index读取，将多个三维数组按通道拼接为一个三维数组
        p = CycleGANDataPreprocessor()
        self.tra = []
        for path in self.paths_tra:
            data = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
            # 通过CycleGANDataPreprocessor获取转置方向
            temp = p.get_norm_target(path.replace('.nii.gz', '.npy'))
            axes = p.norm_direction(data, temp)
            data = np.transpose(data, axes)
            self.tra.extend(data)

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        p.plot_image(self.tra[100])
        p.plot_image(self.tra[150])
        p.plot_image(self.tra[300])
        p.plot_image(self.tra[400])
        self.trb = []
        for path in self.paths_trb:
            data = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
            # 通过CycleGANDataPreprocessor获取转置方向
            temp = p.get_norm_target(path.replace('.nii.gz', '.npy'))
            axes = p.norm_direction(data, temp)
            data = np.transpose(data, axes)
            self.trb.extend(data)
        p.plot_image(self.trb[100])
        p.plot_image(self.trb[150])
        p.plot_image(self.trb[300])
        p.plot_image(self.trb[400])
        # 分别计算CT和MRI的mean和std
        self.mean_a = np.mean(self.tra)
        self.std_a = np.std(self.tra)
        self.mean_b = np.mean(self.trb)
        self.std_b = np.std(self.trb)
        # 数据transform
        self.transform_A = [transforms.ToTensor()]
        self.transform_B = [transforms.ToTensor()]
        if opt.resize:
            self.transform_A.append(transforms.Resize([opt.load_size, opt.load_size]))
            self.transform_B.append(transforms.Resize([opt.load_size, opt.load_size]))
        if opt.flip:
            self.transform_A.append(transforms.RandomHorizontalFlip())
            self.transform_B.append(transforms.RandomHorizontalFlip())
        if opt.norm:
            self.transform_A.append(transforms.Normalize((self.mean_a,), (self.std_a,)))
            self.transform_B.append(transforms.Normalize((self.mean_b,), (self.std_b,)))
        if not opt.resize:
            self.transform_A.append(letterbox_image(opt.load_size))
            self.transform_B.append(letterbox_image(opt.load_size))
        self.transform_A = transforms.Compose(self.transform_A)
        self.transform_B = transforms.Compose(self.transform_B)
        # 数据transform
        for i in range(len(self)):
            self.tra[i] = self.transform_A(self.tra[i])
            self.trb[i] = self.transform_B(self.trb[i])

    def __getitem__(self, index):
        # 读取.niigz并transform，一次返回一整个3D图像
        arr_a = self.tra[index]
        arr_b = self.trb[index]
        return {
            'A': arr_a,
            'B': arr_b,
            'A_paths': "",
            'B_paths': ""
        }

    def __len__(self):
        return min(len(self.tra), len(self.trb))


if __name__ == '__main__':
    img = np.zeros((30, 100, 100), dtype=np.float32)
    img[:, :, :] = 255
    img[:, 10:90, 10:90] = 500
    plt.imshow(img[0], cmap='gray')
    plt.show()
    img = torch.from_numpy(img).cuda()
    plt.imshow(letterbox_image(img, (200, 150)).cpu().numpy()[0], cmap='gray')
    plt.show()
    plt.imshow(reverse_letterbox_image(letterbox_image(img, (200, 150)), (100, 100))[0], cmap='gray')
    plt.show()
