# 将extend_train下的mri改为ct
import os

dir_path = 'extend_train'

files = os.listdir(dir_path)

# 先改image
files_img = [i for i in files if 'image' in i]
for i in range(len(files_img)):
    file = files_img[i]
    cnt = str(1000+i+1)
    os.rename(os.path.join(dir_path, file), os.path.join(dir_path, file.replace('mri', 'ct').replace(cnt, str(1020+i+1))))

files_label = [i for i in files if 'label' in i]
for i in range(len(files_label)):
    file = files_label[i]
    cnt = str(1000+i+1)
    os.rename(os.path.join(dir_path, file), os.path.join(dir_path, file.replace('mri', 'ct').replace(cnt, str(1020+i+1))))

