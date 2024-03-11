import numpy as np
import matplotlib.pyplot as plt
import hashlib
import SimpleITK as sitk

# 读取文件分析后，确定nnUNetPlans_2d目录下.npy为处理后的输入,
# seg.npy为处理后的标签
# .npz为将输入和标签压缩保存后的结果
# 具体保存方法为 np.savez_compressed(保存路径, data=npy, seg=seg_npy)
def calculate_sha256(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
        sha256_hash = hashlib.sha256(content)
        return sha256_hash.hexdigest()


if __name__ == '__main__':
    # .npy
    npy_path = '../nnUNet_Data/nnUNet_preprocessed/Dataset602_MMWHS2017_CT/nnUNetPlans_2d/ct_train_1001.npy'
    npy = np.load(npy_path)
    plt.imshow(npy[0][150])
    plt.show()

    # seg.npy
    seg_path = '../nnUNet_Data/nnUNet_preprocessed/Dataset602_MMWHS2017_CT/nnUNetPlans_2d/ct_train_1001_seg.npy'
    seg_npy = np.load(seg_path)
    plt.imshow(seg_npy[0][150])
    plt.show()

    # .npy保存为'data',seg.npy保存为'seg'
    np.savez_compressed(seg_path.replace('.npy', '.npz'), data=npy, seg=seg_npy)

    # .npz
    npz_path = '../nnUNet_Data/nnUNet_preprocessed/Dataset602_MMWHS2017_CT/nnUNetPlans_2d/ct_train_1001.npz'
    npz = np.load(npz_path)
    key = 'data'
    print(np.all(np.equal(npz[key][0][150], npy[0][150])))
    plt.imshow(npz[key][0][150])
    plt.show()

    # 读取先前保存的.npy，计算sha256值，与读取的.npz计算的hash值进行比较
    hash_seg_npz = calculate_sha256(seg_path.replace('.npy', '.npz'))
    hash_npz = calculate_sha256(npz_path)
    print(hash_seg_npz == hash_npz) # 相同
