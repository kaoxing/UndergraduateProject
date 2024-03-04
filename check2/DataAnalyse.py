import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from collections import Counter
import time


def get_data_list(path):
    data_list = []
    img_list = os.listdir(path)
    readed_list = []
    for i in range(len(img_list)):
        if img_list[i] in readed_list:
            continue
        readed_list.append(img_list[i])
        readed_list.append(img_list[i].replace("image", "label"))
        img = sitk.ReadImage(os.path.join(path, img_list[i]))
        label = sitk.ReadImage(os.path.join(path, img_list[i].replace("image", "label")))
        data_list.append([img, label])
    return data_list


# 统计数据集中的数据的均值和标准差
def get_data_mean_std(path):
    data_list = get_data_list(path)
    img_list = []
    for data in tqdm(data_list):
        img = sitk.GetArrayFromImage(data[0])
        img_list.append(img.flatten())
    img_list = np.array(img_list)
    mean = np.mean(img_list)
    std = np.std(img_list)


# 绘制心脏目标区域的频率直方图
def draw_heart_histogram(path):
    data_list = get_data_list(path)
    # 统计label中不同值的数量
    # 用Counter的方法统计
    # start = time.perf_counter()
    # for data in data_list:
    #     label = sitk.GetArrayFromImage(data[1])
    #     print(Counter(label.flatten()))
    # end = time.perf_counter()
    # print("Cost Time: ", end - start)

    # 统计并绘制img中不同值的数量，并归一化
    # 用Counter的方法统计
    heart_list = []
    for data in tqdm(data_list):
        img = sitk.GetArrayFromImage(data[0])
        heart_list.extend(img.flatten())
    data = Counter(heart_list)
    data[0] = 0  # 将背景像素值0的数量置为0
    x = list(data.keys())
    y = [i / sum(data.values()) for i in list(data.values())]

    plt.xlabel("Value")  # x轴表示像素值
    plt.ylabel("Frequency")  # y轴表示像素值出现的频率
    plt.title("Frequency Histogram")
    plt.plot(x, y, color="red")
    plt.show()

    # plt.hist(heart_list, bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("Heart")
    # plt.ylabel("Frequency")
    # plt.title("Heart Frequency Histogram")
    # plt.show()




if __name__ == '__main__':
    draw_heart_histogram()
