import os
import numpy as np
import multiprocessing
import skimage
import SimpleITK as sitk


# 标签边缘value为0，向内部value逐层递增，向外部value逐层递减，以实现类似山峰的效果

label_dir = '../nnUNet_Data/nnUNet_raw/Dataset602_MMWHS2017_CT/labelsTr'

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
    # 将六种类型的边缘合并
    label_edge = lv_edge | rv_edge | la_edge | ra_edge | myo_edge | aa_edge
    ret = np.zeros(label_edge.shape, dtype=np.uint16)
    ret[label_edge] = 1
    return ret

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
        label_edges = []
        for img in imgs_array:
            label_edges.append(task(img))
        label_edges = np.array(label_edges)
        label_edges = sitk.GetImageFromArray(label_edges)
        label_edges.CopyInformation(imgs)
        sitk.WriteImage(label_edges, path.replace('labelsTr', 'labelsTr_edge'))
        # time.sleep(1)



def boundaries(p_num):
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
    print("temp_paths:", temp_paths)


if __name__ == '__main__':
    boundaries(4)
