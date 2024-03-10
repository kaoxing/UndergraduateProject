# 将MMWHS2017数据集转换为nnUNet的数据集格式

# export nnUNet_raw="/root/nnUNet/nnUNet_data/nnUNet_raw"
# export nnUNet_preprocessed="/root/nnUNet/nnUNet_data/nnUNet_preprocessed"
# export nnUNet_results="/root/nnUNet/nnUNet_data/nnUNet_results"

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from boundaries import boundaries
from merge_boundaries_with_seg import merge_boundaries_with_seg

from check2.generate_dataset_json import generate_dataset_json

nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')
dataset_name = f"Dataset603_MMWHS2017_CT"
out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
out_train_dir = out_dir / "imagesTr"
out_labels_dir = out_dir / "labelsTr"
out_test_dir = out_dir / "imagesTs"
mr_train_dir = f"../RawData/MM-WHS 2017 Dataset/mr_train"
mr_test_dir = f"../RawData/MM-WHS 2017 Dataset/mr_test"
ct_test_dir = f"../RawData/MM-WHS 2017 Dataset/ct_test"
ct_train_dir = f"../RawData/MM-WHS 2017 Dataset/ct_train"


def make_out_dirs():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files():
    """Copy files from the MMWHS dataset to the nnUNet dataset folder."""

    # copy mri training files
    for file in tqdm(os.listdir(ct_train_dir)):
        if 'image' in file:
            shutil.copy(
                os.path.join(ct_train_dir, file),
                os.path.join(out_train_dir, file.replace("_image", "_0000"))
            )
        elif 'label' in file:
            shutil.copy(
                os.path.join(ct_train_dir, file),
                os.path.join(out_labels_dir, file.replace("_label", ""))
            )

    # copy mri test files
    for file in tqdm(os.listdir(ct_test_dir)):
        if 'image' in file:
            shutil.copy(
                os.path.join(ct_test_dir, file),
                os.path.join(out_test_dir, file.replace("_image", "_0000"))
            )


# change MMWHS2017 labels to nnUNet labels
def modify_label_value():
    for file in tqdm(Path(out_labels_dir).iterdir()):
        label = sitk.ReadImage(file)
        label_array = sitk.GetArrayFromImage(label)
        label_array[label_array == 500] = 1
        label_array[label_array == 600] = 2
        label_array[label_array == 420] = 3
        label_array[label_array == 550] = 4
        label_array[label_array == 205] = 5
        label_array[label_array == 820] = 6
        label_array[label_array > 6] = 0
        label_new = sitk.GetImageFromArray(label_array)
        label_new.CopyInformation(label)
        sitk.WriteImage(label_new, file)


def generate_json():
    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "LV": 1,
            "RV": 2,
            "LA": 3,
            "RA": 4,
            "Myo": 5,
            "AA": 6,
            "LV_edges": 7,
            "RV_edges": 8,
            "LA_edges": 9,
            "RA_edges": 10,
            "Myo_edges": 11,
            "AA_edges": 12,
        },
        file_ending=".nii.gz",
        num_training_cases=20,
        overwrite_image_reader_writer="SimpleITKImageIO",
    )


if __name__ == '__main__':
    make_out_dirs()
    copy_files()
    modify_label_value()
    generate_json()

    # 生成边缘标签
    print("Start to generate boundaries....")
    boundaries(6, out_labels_dir)

    # 将边缘标签合并到原始标签中
    print("Start to merge boundaries with seg....")
    merge_boundaries_with_seg(out_labels_dir, out_labels_dir.replace('labelsTr', 'labelsTr_edge'))
    print("All done.")

