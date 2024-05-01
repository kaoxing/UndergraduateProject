# author:kaoxing
# 心脏影像分割系统
import sys

from PyQt5 import QtWidgets

from GUI.UI.untitled import Ui_Form

class AppWindow(Ui_Form,QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # self.ui = Ui_Form()
        self.setupUi(self)
        self.show()







if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = AppWindow()
    application.show()
    sys.exit(app.exec())
