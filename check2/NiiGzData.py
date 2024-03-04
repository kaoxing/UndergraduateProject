import SimpleITK as sitk


class NiiGzData:
    """用于处理NiiGz格式数据的类"""

    def __init__(self, path):
        self.path = path
        self.img = sitk.ReadImage(self.path)
        self.data = sitk.GetArrayFromImage(self.img)
        self.spacing = self.img.GetSpacing()
        self.origin = self.img.GetOrigin()
        self.direction = self.img.GetDirection()
        self.size = self.data.shape

    def get_data(self):
        return self.data

    def get_spacing(self):
        return self.spacing

    def get_origin(self):
        return self.origin

    def get_direction(self):
        return self.direction

    def get_size(self):
        return self.size

    def get_path(self):
        return self.path

    def get_img(self):
        return self.img

    def set_data(self, data):
        self.data = data

    def set_spacing(self, spacing):
        self.spacing = spacing

    def set_origin(self, origin):
        self.origin = origin

    def set_direction(self, direction):
        self.direction = direction

    def set_size(self, size):
        self.size = size

    def set_path(self, path):
        self.path = path

    def set_img(self, img):
        self.img = img

    def save_data(self, path):
        self.img = sitk.GetImageFromArray(self.data)
        self.img.SetSpacing(self.spacing)
        self.img.SetOrigin(self.origin)
        self.img.SetDirection(self.direction)
        sitk.WriteImage(self.img, path)

    def save_img(self, path):
        sitk.WriteImage(self.img, path)

    def save_data_as_nii(self, path):
        self.img = sitk.GetImageFromArray(self.data)
        self.img.SetSpacing(self.spacing)
        self.img.SetOrigin(self.origin)
        self.img.SetDirection(self.direction)
        sitk.WriteImage(self.img, path)

    def save_img_as_nii(self, path):
        sitk.WriteImage(self.img, path)

    def save_data_as_nii_gz(self, path):
        self.img = sitk.GetImageFromArray(self.data)
        self.img.SetSpacing(self.spacing)
        self.img.SetOrigin(self.origin)
        self.img.SetDirection(self.direction)
        sitk.WriteImage(self.img, path)

    def save_img_as_nii_gz(self, path):
        sitk.WriteImage(self.img, path)


