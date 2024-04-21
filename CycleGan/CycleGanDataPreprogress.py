# author:kaoxing
# date: 2024/4/21
import multiprocessing
import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
import tqdm


class CycleGANDataPreprocessor:
    def __init__(self):
        self.size = 488
        self.ct_result_dir = 'CimageTr'
        self.label_result_dir = 'ClabelTr'
        self.ct_dir = r'D:\pythonProject\UndergraduateProject\data\nnUNet_data\nnUNet_raw\Dataset602_MMWHS2017_CT'
        self.train_name = 'imagesTr'
        self.label_name = 'labelsTr'
        self.mri_dir = r'D:\pythonProject\UndergraduateProject\data\nnUNet_data\nnUNet_raw\Dataset601_MMWHS2017_MRI'

    def make_dir(self):
        """
        创建文件夹
        """
        os.makedirs(os.path.join(self.ct_dir, self.ct_result_dir), exist_ok=True)
        os.makedirs(os.path.join(self.ct_dir, self.label_result_dir), exist_ok=True)

    def remove_image_no_label(self, images, labels):
        """
        移除标签全为0的图像，由于标签是连续的，因此只要找到第一张出现标签的图像和其后第一张未出现标签的图像即可
        :param images:
        :param labels:
        :return:
        """
        start = 0
        end = 0
        for i in range(len(labels)):
            if np.max(labels[i]) > 0:
                start = i
                break
        for i in range(start, len(labels)):
            if np.max(labels[i]) == 0:
                end = i
                break
        # 切片
        return images[start:end], labels[start:end]

    def find_labels_max_bbox(self, labels):
        """
        找到一个满足所有label的最大bbox
        :param label: label列表,label为ndarray
        :return: bbox
        """
        bbox = [9999, 9999, 0, 0]
        for label in labels:
            img = Image.fromarray(label)
            img = img.convert('L')
            img = img.point(lambda p: p > 0)
            q = img.getbbox()
            bbox[0] = min(bbox[0], q[0])
            bbox[1] = min(bbox[1], q[1])
            bbox[2] = max(bbox[2], q[2])
            bbox[3] = max(bbox[3], q[3])
        return bbox

    def enlarge_bbox(self, raw_size, bbox, enlarge=1.1):
        """
        将bbox扩大enlarge倍,且不超过raw_size
        :param raw_size: 图片原始大小
        :param bbox: bbox
        :param enlarge: 扩大倍数
        :return: 扩大后的bbox
        """
        # 扩大bbox
        bbox = [int(bbox[0] - (bbox[2] - bbox[0]) * (enlarge - 1) / 2),
                int(bbox[1] - (bbox[3] - bbox[1]) * (enlarge - 1) / 2),
                int(bbox[2] + (bbox[2] - bbox[0]) * (enlarge - 1) / 2),
                int(bbox[3] + (bbox[3] - bbox[1]) * (enlarge - 1) / 2)]
        # 裁剪bbox
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(raw_size[1], bbox[2])
        bbox[3] = min(raw_size[0], bbox[3])

    def crop_images(self, image, bbox):
        """
        按照bbox裁剪图片
        :param image: 图片
        :param bbox: bbox
        :param enlarge: 扩大比例
        :return: 裁剪后的图片
        """
        img = [Image.fromarray(i).crop(bbox) for i in image]
        return np.asarray(img)

    def reshape_images(self, image, size):
        """
        将图片reshape为size大小
        :param image: 图片 list ndarray
        :param size: 目标大小
        :return: reshape后的图片
        """
        img = [Image.fromarray(i).resize(size) for i in image]
        return np.asarray(img)

    def plot_image(self, image):
        """
        绘制图片
        :param image: 图片 ndarray
        """
        import matplotlib.pyplot as plt
        plt.imshow(image, cmap='gray')
        plt.show()

    def get_new_spacing(self, old_spacing, old_size, new_size):
        """
        计算新的spacing
        :param old_spacing: 旧spacing
        :param old_size: 旧大小
        :param new_size: 新大小
        """
        new_spacing = [old_spacing[i] * old_size[i] / new_size[i] for i in range(3)]
        return new_spacing

    def task(self, ct_paths, lock, i):
        # 读取ct图和标签
        while True:
            lock.acquire()
            if len(ct_paths) == 0:
                lock.release()
                break
            ct_path = ct_paths.pop()
            lock.release()
            print(f"I am {i}, processing:", ct_path)
            ct = sitk.ReadImage(ct_path)
            ct_label = sitk.ReadImage(ct_path.replace(self.train_name, self.label_name).replace("_0000", ""))
            ct_array = sitk.GetArrayFromImage(ct)
            ct_label_array = sitk.GetArrayFromImage(ct_label)
            # sitk读取的数据为z,y,x,将其转为y,z,x
            ct_array = np.transpose(ct_array, (1, 0, 2))
            ct_label_array = np.transpose(ct_label_array, (1, 0, 2))
            # 移除标签全为0的图像
            ct_array, ct_label_array = self.remove_image_no_label(ct_array, ct_label_array)
            # 获取label的bbox
            bbox = self.find_labels_max_bbox(ct_label_array)
            # 扩大bbox
            self.enlarge_bbox(ct_array.shape[1:], bbox)
            # 裁剪ct和label
            ct_array = self.crop_images(ct_array, bbox)
            ct_label_array = self.crop_images(ct_label_array, bbox)
            crop_size = ct_array.shape
            # reshape
            ct_array = self.reshape_images(ct_array, (self.size, self.size))
            ct_label_array = self.reshape_images(ct_label_array, (self.size, self.size))
            # plot_image(ct_array)
            # 将数据转为原本的z,y,x
            ct_array = np.transpose(ct_array, (1, 0, 2))
            ct_label_array = np.transpose(ct_label_array, (1, 0, 2))
            # 计算新的spacing
            new_spacing = self.get_new_spacing(ct.GetSpacing(), crop_size, (self.size, self.size, crop_size[2]))
            # 创建新Image修改info
            new_ct = sitk.GetImageFromArray(ct_array)
            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetSpacing(new_spacing)
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct_label = sitk.GetImageFromArray(ct_label_array)
            new_ct_label.SetDirection(ct_label.GetDirection())
            new_ct_label.SetSpacing(new_spacing)
            new_ct_label.SetOrigin(ct_label.GetOrigin())
            # 保存
            sitk.WriteImage(new_ct, os.path.join(self.ct_dir, self.ct_result_dir, os.path.basename(ct_path)))
            sitk.WriteImage(new_ct_label, os.path.join(self.ct_dir, self.label_result_dir, os.path.basename(
                ct_path.replace(self.train_name, self.label_name).replace("_0000", ""))))

    def handle_CycleGAN(self, p_num):
        self.make_dir()
        ct_paths = [os.path.join(self.ct_dir, self.train_name, i) for i in os.listdir(os.path.join(self.ct_dir,
                                                                                                   self.train_name))]
        share_paths = multiprocessing.Manager().list(ct_paths)
        lock = multiprocessing.Manager().Lock()
        # 多进程处理
        processes = []
        for i in range(p_num):
            p = multiprocessing.Process(target=self.task, args=(share_paths, lock, i))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


if __name__ == '__main__':
    # 处理CT和MRI数据作为有效的cycleGan数据集
    p = CycleGANDataPreprocessor()
    p.make_dir()
    p.handle_CycleGAN(4)
    # 修改ct_dir为mri_dir
    p.ct_dir = p.mri_dir
    p.make_dir()
    p.handle_CycleGAN(4)
    print('done')
