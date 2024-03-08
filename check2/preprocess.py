# author kaoxing

# 此脚本搭配resample_data_or_seg_to_shape使用，用于将原始标签数据进行重采样，以符合nnUNet的数据
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
from nnUNet.nnUNet.nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

edges_path = '../nnUNet_Data/nnUNet_raw/Dataset602_MMWHS2017_CT/labelsTr_edge'

# data_shape = [1, 363, 512, 512]
new_shape = [363, 415, 415]
current_spacing = [0.44999998807907104, 0.35546875, 0.35546875]
new_spacing = [0.44999998807907104, 0.4384765028953552, 0.4384765028953552]
is_seg = True
order = 1
order_z = 0
force_separate_z = None
separate_z_anisotropy_threshold = 3


# 多进程处理
def task(temp_paths, lock, i):
    while True:
        lock.acquire()
        if len(temp_paths) == 0:
            lock.release()
            break
        path = temp_paths.pop()
        print(f"I am {i}, processing:", path)
        lock.release()
        edge = np.load(path)['data']
        # plt.imshow(np.load(edges_paths[0])['data'][150][0], cmap='gray')
        # plt.show()
        # 重组数据，将[363,6,512,512]的数据重组为[6,1,363,512,512]
        edge = edge[np.newaxis, ...]  # [1, 363, 6, 512, 512]
        edge = edge.transpose((2, 0, 1, 3, 4))  # [6, 1, 363, 512, 512]
        # plt.imshow(edge[0][0][150], cmap='gray')
        # plt.show()
        # 重采样，输入[1, 363, 512, 512]的数据
        new_data = []
        for data in edge:
            new_data.append(
                resample_data_or_seg_to_shape(data, new_shape, is_seg, current_spacing, new_spacing, order, order_z,
                                              force_separate_z, separate_z_anisotropy_threshold))
        # plt.imshow(new_data[0][0][150], cmap='gray')
        # plt.show()
        # 保存为npz文件
        np.savez_compressed(
            path.replace('nnUNet_raw', 'nnUNet_preprocessed').replace('labelsTr_edge', 'gt_segmentations_edges'),
            data=new_data)


def preprocess(p_nums):
    edges_names = os.listdir(edges_path)
    edges_paths = [os.path.join(edges_path, edges_name) for edges_name in edges_names]
    # 移除非npz文件
    for edge_path in edges_paths:
        if not edge_path.endswith('.npz'):
            edges_paths.remove(edge_path)
    temp_paths = multiprocessing.Manager().list(edges_paths)
    lock = multiprocessing.Manager().Lock()
    # 多进程处理
    processes = []
    for i in range(p_nums):
        p = multiprocessing.Process(target=task, args=(temp_paths, lock, i))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("All processes are done.")


if __name__ == '__main__':
    t = time.time()
    preprocess(2)
    print("cost:", time.time() - t)
