import os
import time

import numpy as np
import multiprocessing
import skimage
import SimpleITK as sitk
import matplotlib.pyplot as plt

# 标签边缘value为0，向内部value逐层递增，向外部value逐层递减，以实现类似山峰的效果




def grow_inner_edge(img):
    # 将提取出的边缘扩张的更粗
    ret = np.zeros(img.shape, dtype=np.uint16)
    # 循环实现
    # for i in range(1, img.shape[0] - 1):
    #     for j in range(1, img.shape[1] - 1):
    #         if img[i, j] == 1:
    #             ret[i - 1:i + 2, j - 1:j + 2] = 1
    # 高效实现
    ret[1:-1, 1:-1] = img[:-2, :-2] | img[:-2, 1:-1] | img[:-2, 2:] | img[1:-1, :-2] | img[1:-1, 1:-1] | img[1:-1, 2:] | \
                        img[2:, :-2] | img[2:, 1:-1] | img[2:, 2:]
    # 转为bool类型
    ret = ret.astype(bool)
    return ret


def task(img):
    # 将每种类别单独提取为一副图像
    # lv类型的标签值为1
    lv = np.ndarray(img.shape, dtype=bool)
    lv[...] = False
    lv[img == 1] = True
    # rv类型的标签值为2
    rv = np.ndarray(img.shape, dtype=bool)
    rv[...] = False
    rv[img == 2] = True
    # la类型的标签值为3
    la = np.ndarray(img.shape, dtype=bool)
    la[...] = False
    la[img == 3] = True
    # ra类型的标签值为4
    ra = np.ndarray(img.shape, dtype=bool)
    ra[...] = False
    ra[img == 4] = True
    # myo类型的标签值为5
    myo = np.ndarray(img.shape, dtype=bool)
    myo[...] = False
    myo[img == 5] = True
    # aa类型的标签值为6
    aa = np.ndarray(img.shape, dtype=bool)
    aa[...] = False
    aa[img == 6] = True
    # 六种类型的标签分别提取出来
    lv_edge = skimage.segmentation.find_boundaries(lv)
    rv_edge = skimage.segmentation.find_boundaries(rv)
    la_edge = skimage.segmentation.find_boundaries(la)
    ra_edge = skimage.segmentation.find_boundaries(ra)
    myo_edge = skimage.segmentation.find_boundaries(myo)
    aa_edge = skimage.segmentation.find_boundaries(aa)
    # 将边缘扩张的更粗
    lv_edge = grow_inner_edge(lv_edge)
    rv_edge = grow_inner_edge(rv_edge)
    la_edge = grow_inner_edge(la_edge)
    ra_edge = grow_inner_edge(ra_edge)
    myo_edge = grow_inner_edge(myo_edge)
    aa_edge = grow_inner_edge(aa_edge)
    # 将六种类型的边缘合并，以供检查
    label_edge = lv_edge | rv_edge | la_edge | ra_edge | myo_edge | aa_edge
    ret1 = np.zeros(label_edge.shape, dtype=np.uint16)
    ret1[label_edge] = 1
    # 六种不同类型的边缘分别存储
    ret2 = [lv_edge.astype(np.uint16), rv_edge.astype(np.uint16), la_edge.astype(np.uint16), ra_edge.astype(np.uint16),
            myo_edge.astype(np.uint16), aa_edge.astype(np.uint16)]
    return ret1, ret2


def get_boundaries(temp_paths, lock, i):
    while True:
        lock.acquire()
        if len(temp_paths) == 0:
            lock.release()
            break
        path = temp_paths.pop()
        print(f"I am {i}, processing:", path)
        lock.release()
        imgs = sitk.ReadImage(path)
        imgs_array = sitk.GetArrayFromImage(imgs)
        edges_merge = []
        edges_divide = []
        for img in imgs_array:
            edge_merge, edge_divide = task(img)
            edges_merge.append(edge_merge)
            edges_divide.append(edge_divide)
        # 保存合并的边缘图像
        edges_merge = np.array(edges_merge)
        edges_merge = sitk.GetImageFromArray(edges_merge)
        edges_merge.CopyInformation(imgs)
        sitk.WriteImage(edges_merge, path.replace('labelsTr', 'labelsTr_edge').replace('.nii.gz', 'bigger.nii.gz'))
        # 以npz形式保存分开的边缘图像
        edges_divide = np.array(edges_divide)
        np.savez_compressed(path.replace('labelsTr', 'labelsTr_edge').replace('.nii.gz', '.npz'), data=edges_divide)


def boundaries(p_num, label_dir):
    os.makedirs(label_dir.replace('labelsTr', 'labelsTr_edge'), exist_ok=True)
    label_names = os.listdir(label_dir)
    label_paths = [os.path.join(label_dir, label_name) for label_name in label_names]
    share_paths = multiprocessing.Manager().list(label_paths)
    lock = multiprocessing.Manager().Lock()
    temp_paths = label_paths.copy()
    # 多进程处理
    processes = []
    for i in range(p_num):
        p = multiprocessing.Process(target=get_boundaries, args=(share_paths, lock, i))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == '__main__':
    t = time.time()
    boundaries(6, label_dir)
    print("cost:", time.time() - t)

    # 读取npz文件并用matplotlib显示
    # import matplotlib.pyplot as plt
    # import numpy as np
    # file_path = '../nnUNet_Data/nnUNet_raw/Dataset602_MMWHS2017_CT/labelsTr_edge/ct_train_1001.npz'
    # data = np.load(file_path)
    # key = 'arr_150'
    # temp = data[key][0]
    # plt.imshow(data[key][0], cmap='gray')
    # plt.show()
    # plt.imshow(data[key][1], cmap='gray')
    # plt.show()
    # plt.imshow(data[key][2], cmap='gray')
    # plt.show()
    # plt.imshow(data[key][3], cmap='gray')
    # plt.show()
    # plt.imshow(data[key][4], cmap='gray')
    # plt.show()
    # plt.imshow(data[key][5], cmap='gray')
    # plt.show()
    # file_path = '../nnUNet_Data/nnUNet_raw/Dataset602_MMWHS2017_CT/labelsTr_edge/ct_train_1001.nii.gz'
    # data = sitk.ReadImage(file_path)
    # data_array = sitk.GetArrayFromImage(data)
    # plt.imshow(data_array[150], cmap='gray')
    # plt.show()
    # file_path = '../nnUNet_Data/nnUNet_raw/Dataset602_MMWHS2017_CT/imagesTr/ct_train_1001_0000.nii.gz'
    # data = sitk.ReadImage(file_path)
    # data_array = sitk.GetArrayFromImage(data)
    # plt.imshow(data_array[150], cmap='gray')
    # plt.show()
    # file_path = '../nnUNet_Data/nnUNet_preprocessed/Dataset602_MMWHS2017_CT/gt_segmentations/ct_train_1001.nii.gz'
    # data = sitk.ReadImage(file_path)
    # data_array = sitk.GetArrayFromImage(data)
    # plt.imshow(data_array[150], cmap='gray')
    # plt.show()
