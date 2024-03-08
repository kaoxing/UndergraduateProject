# 将边缘作为额外的分类类别，加入原始分割标签中
import os

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
# 边缘标签value

seg_classes = 6
edges_value = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6} # 1: lv, 2: rv, 3: la, 4: ra, 5: myo, 6: aa
edges_value = {k: v + seg_classes for k, v in edges_value.items()}


def merge_boundaries_with_seg(path_seg, path_edges):
    # 读取分割标签 .nii.gz文件
    seg = sitk.ReadImage(path_seg)
    seg_array = sitk.GetArrayFromImage(seg)  # [363, 512, 512]
    # 读取边缘标签 .npz文件
    edges_array = np.load(path_edges)['data']  # [363, 6, 512, 512]
    # 将边缘标签加入分割标签中
    for i in range(len(seg_array)):
        temp = seg_array[i]
        edges = edges_array[i]
        for j in range(seg_classes):
            value = j + 1
            temp[edges[j] == 1] = seg_classes+value
    # 保存为.nii.gz文件
    seg_new = sitk.GetImageFromArray(seg_array)
    seg_new.CopyInformation(seg)
    sitk.WriteImage(seg_new, path_seg)




if __name__ == '__main__':
    path_seg = r"..\nnUNet_Data\nnUNet_raw\Dataset602_MMWHS2017_CT\labelsTr"
    path_edges = r"..\nnUNet_Data\nnUNet_raw\Dataset602_MMWHS2017_CT\labelsTr_edge"

    files_seg = os.listdir(path_seg)

    files_edges = os.listdir(path_edges)
    files_edges = [file for file in files_edges if file.endswith('.npz')]

    files_seg = [os.path.join(path_seg, file) for file in files_seg]
    files_edges = [os.path.join(path_edges, file) for file in files_edges]

    for i in range(len(files_seg)):
        merge_boundaries_with_seg(files_seg[i], files_edges[i])
        print(f"{i+1}/{len(files_seg)} is done.")
