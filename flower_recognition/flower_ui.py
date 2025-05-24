# flower_recognition_ui.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from testModel import FlowerRecognitionModel  # 导入模型处理类

class FlowerRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = FlowerRecognitionModel('model/best_model1.pth')  # 加载模型

    def initUI(self):
        self.setWindowTitle('花卉识别系统')
        self.setGeometry(300, 300, 800, 600)

        # 组件定义
        self.btn_open = QPushButton('选择图片', self)
        self.btn_open.clicked.connect(self.open_image)
        self.label_image = QLabel('图片显示区域', self)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_result = QLabel('识别结果将显示在此处', self)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.btn_open)
        layout.addWidget(self.label_image)
        layout.addWidget(self.label_result)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self):
        # 文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp)'
        )
        if file_path:
            self.predict_image(file_path)  # 执行预测

    def predict_image(self, image_path):
        # 预测流程
        predicted_class = self.model.predict(image_path)
        self.display_result(image_path, predicted_class)

    def display_result(self, image_path, class_id):
        # 显示图片和结果
        pixmap = QPixmap(image_path).scaled(400, 400, Qt.KeepAspectRatio)
        self.label_image.setPixmap(pixmap)

        # 假设类别标签字典（需替换为你的实际类别）
        class_names = {
            0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'
            # ... 其他类别
        }
        self.label_result.setText(f'识别结果：{class_names.get(class_id, "未知花卉")}')


# 运行程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FlowerRecognitionApp()
    ex.show()
    sys.exit(app.exec_())
