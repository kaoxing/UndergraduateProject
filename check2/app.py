# author:kaoxing
# 心脏影像分割系统
import os
import shutil
import subprocess
import sys
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter
import SimpleITK as sitk
from PyQt5 import QtWidgets, QtCore, Qt
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from GUI.UI.untitled import Ui_Form
from qt_material import apply_stylesheet

class SegmentationThread(QThread):
    finished = pyqtSignal()

    def __init__(self, raw_path, input_folder, output_folder, exe_path, option):
        super().__init__()
        self.raw_path = raw_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.exe_path = exe_path
        self.option = option

    def run(self):
        shutil.copy(self.raw_path, self.input_folder)
        command = self.exe_path + self.option
        os.system(command)
        self.finished.emit()


class AppWindow(Ui_Form, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer = None
        self.option = None
        self.output_folder = None
        self.input_folder = None
        self.exe_path = None
        self.thread = None
        self.segmented_image = None
        self.setupUi(self)
        self.show()
        self.raw_image = None
        self.raw_data = None
        self.seg_image = None
        self.seg_data = None
        self.raw_path = None

    def load_seg(self, seg_path):
        self.seg_image = sitk.ReadImage(seg_path)
        self.seg_data = sitk.GetArrayFromImage(self.seg_image)

    def add_raw_image(self, index):
        """
        Method to add the raw image to the GraphicsView.
        """
        if index < 0 or index >= len(self.raw_data):
            return None

        img_array = self.raw_data[index]

        if img_array.min() >= img_array.max():
            return None

        img = ((img_array - img_array.min()) * (1 / (img_array.max() - img_array.min()) * 255)).astype('uint8')
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.GraphicsView.width() - 30, self.GraphicsView.height() - 30)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.GraphicsView.setScene(scene)
        return None

    def add_segmentation_result(self, index):
        """
        Method to add the segmentation result to the GraphicsView.
        """
        if index < 0 or index >= len(self.raw_data):
            return None

        img_array = self.seg_data[index]

        if img_array.min() >= img_array.max():
            return None

        img = ((img_array - img_array.min()) * (1 / (img_array.max() - img_array.min()) * 255)).astype('uint8')
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.GraphicsView.width() - 30, self.GraphicsView.height() - 30)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.GraphicsView.setScene(scene)
        return None

    def show_image(self):
        """
        Method to show the image in the GraphicsView.
        """
        pixmap = None

        if self.showRaw.isChecked():
            pixmap = self.add_raw_image(int(self.spinBox.value()))
        elif self.showSegmentation.isChecked():
            seg_pixmap = self.add_segmentation_result(int(self.spinBox.value()))
        else:
            self.GraphicsView.setScene(None)

    def pressImport(self):
        """
        Method to handle the event when the ImportDataButton is clicked.
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "NIfTI Files (*.nii.gz)", options=options)
        if fileName:
            self.raw_image = sitk.ReadImage(fileName)
            self.raw_data = sitk.GetArrayFromImage(self.raw_image)
            self.ImportDataPath.setText(os.path.basename(fileName))
            self.raw_path = fileName
            SpacingX,SpacingY,SpacingZ = self.raw_image.GetSpacing()
            self.SpacingX.setText(str(SpacingX)[:8])
            self.SpacingY.setText(str(SpacingY)[:8])
            self.SpacingZ.setText(str(SpacingZ)[:8])
            ResolutionX,ResolutionY,ResolutionZ = self.raw_image.GetSize()
            self.ResolutionX.setText(str(ResolutionX))
            self.ResolutionY.setText(str(ResolutionY))
            self.ResolutionZ.setText(str(ResolutionZ))

    def pressSegmentation(self):
        """
        Method to handle the event when the StartSegmentation button is clicked.
        """
        # Disable all buttons
        self.StartSegmentation.setEnabled(False)
        self.ImportDataButton.setEnabled(False)
        self.OutputData.setEnabled(False)
        # ... disable other buttons ...

        # Start a timer to update the progress bar
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateProgressBar)
        self.timer.start(1000)  # update every second

        self.exe_path = r"E:\anaconda\envs\torch3.9\Scripts\nnUNetv2_predict.exe "
        self.output_folder = r"D:\pythonProject\UndergraduateProject\data\output"
        self.input_folder = r"D:\pythonProject\UndergraduateProject\data\input"
        self.option = rf"-i {self.input_folder} -o {self.output_folder} -d {602} -d 602 -tr myTrainer -c 2d -f 4"
        # os.remove(input_folder)
        # os.remove(output_folder)
        # os.makedirs(input_folder)
        # os.makedirs(output_folder)
        self.thread = SegmentationThread(self.raw_path, self.input_folder, self.output_folder, self.exe_path,
                                         self.option)
        self.thread.finished.connect(self.onSegmentationFinished)
        self.thread.start()

    def onSegmentationFinished(self):
        # Stop the timer and set the progress bar to 100%
        self.timer.stop()
        self.progressBar.setValue(100)

        # Show a message box when the process is done
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("The process has finished.")
        msg.setWindowTitle("Information")
        msg.exec_()

        # Enable all buttons
        self.StartSegmentation.setEnabled(True)
        self.ImportDataButton.setEnabled(True)
        self.OutputData.setEnabled(True)
        # ... enable other buttons ...
        files = os.listdir(self.output_folder)
        file = [file for file in files if file.endswith('.nii.gz')][0]
        self.load_seg(os.path.join(self.output_folder, file))
        self.show_image()

    def updateProgressBar(self):
        """
        Method to update the progress bar.
        """
        value = self.progressBar.value()
        if value < 100:
            self.progressBar.setValue(value + 1)
        else:
            self.timer.stop()

    def pressExport(self):
        """
        Method to handle the event when the OutputData button is clicked.
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "NIfTI Files (*.nii.gz)", options=options)
        if fileName:
            files = os.listdir(self.output_folder)
            file = [file for file in files if file.endswith('.nii.gz')][0]
            shutil.copy(os.path.join(self.output_folder, file), fileName)
            # 删除原文件
            os.remove(os.path.join(self.output_folder, file))

    def showRawImage(self, state):
        """
        Method to handle the event when the state of checkBox_2 changes.
        """
        if state == QtCore.Qt.Checked:
            self.showSegmentation.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.showSegmentation.setCheckState(QtCore.Qt.Checked)
        self.show_image()

    def showSegmentationResult(self, state):
        """
        Method to handle the event when the state of checkBox changes.
        """
        if state == QtCore.Qt.Checked:
            self.showRaw.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.showRaw.setCheckState(QtCore.Qt.Checked)
        self.show_image()

    def switchImage(self, value):
        """
        Method to handle the event when the value of spinBox changes.
        """
        self.show_image()

    def loadCheckPoint(self, index):
        """
        Method to handle the event when a cell in TableView is double clicked.
        """
        pass  # Add your code here


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    application = AppWindow()
    apply_stylesheet(app, theme='dark_medical.xml')
    application.show()
    sys.exit(app.exec())
