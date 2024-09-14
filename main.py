import sys,os,pickle
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, \
                            QLabel, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from get_face_feature import save_feature
from detect_face_align_rec import get_reconginzed_face

class FaceRecognitionSystem(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle("人脸识别系统")
        self.initUI()

    def initUI(self):
        # 垂直布局（主布局）
        main_layout = QVBoxLayout()

        # 标题标签
        title_label = QLabel("人脸识别系统")
        title_label.setAlignment(Qt.AlignCenter)  # 居中
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # 数据加载区域
        load_layout = QVBoxLayout()
        load_layout.setAlignment(Qt.AlignCenter)

        # 加载pkl文件按钮
        file_load_layout = QHBoxLayout()
        file_load_layout.setAlignment(Qt.AlignCenter)
        load_button = QPushButton("加载数据")
        self.data_path = QLineEdit()
        self.data_path.setPlaceholderText("文本框显示加载数据路径")
        self.data_path.setFocusPolicy(Qt.NoFocus)  # 禁止焦点
        file_load_layout.addWidget(load_button)
        file_load_layout.addWidget(self.data_path)
        load_layout.addLayout(file_load_layout)

        # 添加选择数据库
        db_load_layout = QHBoxLayout()
        db_load_layout.setAlignment(Qt.AlignCenter)
        db_load_button = QPushButton("加载数据库")
        self.db_path = QLineEdit()
        self.db_path.setPlaceholderText("文本框显示数据库路径")
        self.db_path.setFocusPolicy(Qt.NoFocus)  # 禁止焦点
        db_load_layout.addWidget(db_load_button)
        db_load_layout.addWidget(self.db_path)
        load_layout.addLayout(db_load_layout)

        main_layout.addLayout(load_layout)

        # 中部布局，包含两个大方框和中间的按钮
        middle_layout = QHBoxLayout()
        middle_layout.setAlignment(Qt.AlignCenter)  # 居中对齐

        # 左边：选择插入图像区域
        self.image_box = QLabel("选择插入图像")
        self.image_box.setFixedSize(200, 200)
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_box.setStyleSheet("border: 1px solid black;")
        middle_layout.addWidget(self.image_box)

        # 中间：插入按钮和查询按钮
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)  # 居中对齐
        select_button = QPushButton("选择图像")
        query_button = QPushButton("查询插入")
        button_layout.addWidget(select_button)
        button_layout.addWidget(query_button)
        middle_layout.addLayout(button_layout)

        # 右边：数据库中匹配图像区域
        self.match_box = QLabel("数据库中匹配图像")
        self.match_box.setFixedSize(200, 200)
        self.match_box.setAlignment(Qt.AlignCenter)
        self.match_box.setStyleSheet("border: 1px solid black;")
        middle_layout.addWidget(self.match_box)

        main_layout.addLayout(middle_layout)

        # 设置主布局居中
        main_layout.setAlignment(Qt.AlignCenter)

        # 绑定按钮事件
        load_button.clicked.connect(self.load_data)
        db_load_button.clicked.connect(self.load_database)
        select_button.clicked.connect(self.insert_image)
        query_button.clicked.connect(self.query_image)

        self.setLayout(main_layout)

    def load_data(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if folder_path:
            self.data_path.setText(f"load folder: {os.path.basename(folder_path)}") 
            res_path=save_feature(folder_path) 
            if res_path.endswith('.pkl'):
                    with open(res_path, 'rb') as f:
                        dataset = pickle.load(f)
                    self.faces_feature, self.names_list,self.img_cont = dataset
                    self.db_file=res_path
        else:
            QMessageBox.warning(self, "警告", "请选择 .pkl 文件！")

    def load_database(self):
        # 选择数据库路径
        self.db_file, _ = QFileDialog.getOpenFileName(self, "选择数据库", "", "Database Files (*.db *.sqlite *.pkl)")
        if self.db_file:
            self.db_path.setText(f"load pkl file: {os.path.basename(self.db_file)}")
            with open(self.db_file, 'rb') as f:
                dataset = pickle.load(f)
                self.faces_feature, self.names_list,self.img_cont = dataset

    def opencv_to_pixmap(self,image):
        # 检查图像的通道数并转换为 RGB 格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # OpenCV 使用 BGR 格式，QImage 使用 RGB 格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channel = image.shape
        bytes_per_line = 3 * width

        # 创建 QImage 对象
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 转换为 QPixmap
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    
    def show_warning_dialog(self):
        # 创建警告对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText("查寻结果匹配失败！\n是否插入数据库？")
        
        # 添加按钮
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        # 显示对话框并获取用户选择
        result = msg_box.exec_()
        
        if result == QMessageBox.Yes:
            ########## 这里可以添加插入数据库的代码 ##########
            with open(self.db_file, 'rb') as f:
                existing_data=pickle.load(f)
                self.faces_feature, self.names_list,self.img_cont = existing_data
            _, encoded_image = cv2.imencode('.jpg', self.src_img)
            binary_image = encoded_image.tobytes()
            new_data=[np.array(self.feature),os.path.basename(self.file_path).replace(".jpg",""),binary_image]
            assert len(existing_data)==3, "existing data error"
            if new_data[1] not in existing_data[1]:  # 避免重复插入
                existing_data[0] = np.concatenate((existing_data[0], new_data[0]), axis=0)  # 重新赋值给 arr1
                existing_data[1].append(new_data[1])
                existing_data[2].append(new_data[2])
                self.faces_feature, self.names_list,self.img_cont = existing_data[0],existing_data[1],existing_data[2]  #generate new feature
                with open(self.db_file, 'wb') as f:
                    pickle.dump(existing_data, f)
                tmp_msg="{}已插入数据库！".format(new_data[1])
                QMessageBox.information(self, "插入数据库结果", tmp_msg)
                pixmap = QPixmap(self.opencv_to_pixmap(self.src_img))
                scaled_pixmap = pixmap.scaled(self.match_box.width(), self.match_box.height(), Qt.KeepAspectRatio)
                self.match_box.setPixmap(scaled_pixmap)
        else:
            msg_box.close()


    def insert_image(self):
        # 选择要插入的图像
        self.file_path, _ = QFileDialog.getOpenFileName(self, "选择插入图像", "", "Images (*.png *.jpg *.bmp)")
        if self.file_path:
            # 加载图像并显示在左侧的 QLabel (self.image_box) 中
            self.src_img=cv2.imread(self.file_path)
            pixmap = QPixmap(self.opencv_to_pixmap(self.src_img))
            scaled_pixmap = pixmap.scaled(self.image_box.width(), self.image_box.height(), Qt.KeepAspectRatio)
            self.image_box.setPixmap(scaled_pixmap)

    def query_image(self):
        # 模拟查询数据库中匹配的图像
        res,idx,self.feature=get_reconginzed_face(self.src_img,self.faces_feature,self.names_list)
        # 显示匹配结果
        if res!="unknow":
            tmp_cont=self.img_cont[idx]
            nparr = np.frombuffer(tmp_cont, np.uint8)
            db_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # 显示匹配图像在右侧的 QLabel (self.match_box) 中
            pixmap = QPixmap(self.opencv_to_pixmap(db_img))
            scaled_pixmap = pixmap.scaled(self.match_box.width(), self.match_box.height(), Qt.KeepAspectRatio)
            self.match_box.setPixmap(scaled_pixmap)
            QMessageBox.information(self, "查询结果", f"匹配成功！\n匹配人名：{self.names_list[idx]}")
        else:
            # QMessageBox.warning(self, "查询结果", "匹配失败！")
            self.show_warning_dialog()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())
