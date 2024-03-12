# author: Kaoxing
# date: 2024-3-4

# 计算dice系数


def get_dice(path_gd, path_pred):
    gd = sitk.ReadImage(path_gd)
    pred = sitk.ReadImage(path_pred)
    gd_array = sitk.GetArrayFromImage(gd)
    pred_array = sitk.GetArrayFromImage(pred)
    dice = 2 * np.sum(gd_array * pred_array) / (np.sum(gd_array) + np.sum(pred_array))
    return dice