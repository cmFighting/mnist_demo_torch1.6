# TODO 添加一个图形化界面
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import random
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from train_based_torchvision import Net


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('imgs/面性铅笔.png'))
        self.setWindowTitle('手写数字识别')
        # 加载网络
        self.net = Net()
        self.net.load_state_dict(torch.load("models/mnist_net.pth"))
        self.transform = transforms.Compose(
            # 这里只对其中的一个通道进行归一化的操作
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

        self.resize(800, 600)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("测试样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        self.predict_img_path = "imgs/0.jpg"
        img_init = cv2.imread(self.predict_img_path)
        img_init = cv2.resize(img_init, (400, 400))
        cv2.imwrite('imgs/target.png', img_init)
        self.img_label.setPixmap(QPixmap('imgs/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传数字图像 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 识别手写字体 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' 识 别 结 果 ')
        self.result = QLabel("待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        # right_layout.addSpacing(5)
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用手写数字识别系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('imgs/logoxx.png'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>我的个人主页</a>")
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        # git_img = QMovie('images/')
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页面')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('imgs/面性计算器.png'))
        self.setTabIcon(1, QIcon('imgs/面性本子vg.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg , *.png)')
        print(openfile_name)
        img_name = openfile_name[0]
        if img_name=='':
            pass
        else:
            self.predict_img_path = img_name
            img_init = cv2.imread(self.predict_img_path)
            img_init = cv2.resize(img_init, (400, 400))
            cv2.imwrite('imgs/target.png', img_init)
            self.img_label.setPixmap(QPixmap('imgs/target.png'))

    def predict_img(self):
        # 预测图片
        # 开始预测
        img = Image.open(self.predict_img_path)
        gray_img = img.convert('L')
        img_torch = self.transform(gray_img)
        img_torch = img_torch.view(-1, 1, 28, 28)
        outputs = self.net(img_torch)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
        # print("{}的预测结果是:{}".format(img_path, predicted[0].numpy()))
        result = str(predicted[0].numpy())
        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())



