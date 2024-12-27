import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
import csv

class GuiApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("山老鼠剋星偵測系統")
        self.center_window(600, 600)  # 設定視窗大小並使其居中

        self.init_ui()

        # 初始化計時器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_excel_and_update)

        self.row_index = 0  # 新增 row_index 來追蹤讀取到的行數

    def init_ui(self):
        """初始化用戶界面組件"""
        self.layout = QVBoxLayout()

        # 在最上方顯示系統名字
        self.title_label = QLabel("山老鼠剋星偵測系統", self)
        font = QFont("標楷體", 18)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # 顯示英文標題
        self.subtitle_label = QLabel("The Mountain Rat Terminator System", self)
        self.subtitle_label.setFont(QFont("Times New Roman", 18))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.subtitle_label)

        # 顯示圖片並調整大小
        self.image_label = QLabel(self)
        self.pixmap = QPixmap("123.jpg")  # 載入圖片
        scaled_pixmap = self.pixmap.scaled(500, 400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 加入空白區域使按鈕垂直居中
        self.layout.addStretch(1)

        # 創建開始按鈕
        self.create_start_button()

        # 創建顯示文字的Label
        self.create_start_label()

        # 創建並居中顯示交通燈
        self.create_traffic_light()

        # 創建警告Label
        self.create_alert_label()

        # 組件加入佈局
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.start_label)
        self.layout.addLayout(self.traffic_light_layout)
        self.layout.addWidget(self.alert_label)

        self.setLayout(self.layout)

    def center_window(self, window_width, window_height):
        """將視窗置中顯示於螢幕"""
        screen = QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)

    def create_start_button(self):
        """創建並配置 '開始' 按鈕"""
        self.start_button = QPushButton("Press to Start", self)
        font = QFont("Times New Roman", 14)
        self.start_button.setFont(font)
        self.start_button.clicked.connect(self.on_start_button_click)

    def create_start_label(self):
        """創建並配置顯示文字的Label"""
        self.start_label = QLabel("", self)
        font = QFont("Times New Roman", 18)
        self.start_label.setFont(font)
        self.start_label.setAlignment(Qt.AlignCenter)

    def create_traffic_light(self):
        """創建並配置交通燈的Label（綠色/紅色）"""
        self.traffic_light = QLabel(self)
        self.traffic_light.setFixedSize(50, 50)
        self.traffic_light.setStyleSheet("border-radius: 25px; background-color: gray;")
        self.traffic_light_layout = QHBoxLayout()
        self.traffic_light_layout.addStretch(1)
        self.traffic_light_layout.addWidget(self.traffic_light)
        self.traffic_light_layout.addStretch(1)

    def create_alert_label(self):
        """創建警告訊息的Label"""
        self.alert_label = QLabel("", self)
        font = QFont("Times New Roman", 16)
        self.alert_label.setFont(font)
        self.alert_label.setAlignment(Qt.AlignCenter)

    def on_start_button_click(self):
        """按下按鈕後，開始每秒讀取資料"""
        #self.start_label.setText("Start")
        self.row_index = 0  # Reset row_index to start from the first row
        self.timer.start(1000)  # 每 1000 毫秒執行一次
        self.start_button.setEnabled(False)  # 禁用開始按鈕

    def read_excel_and_update(self):
        """讀取資料並更新交通燈"""
        try:
            with open("class_detection.csv", mode="r") as f:
                reader = csv.reader(f)
                lines = list(reader)

            # 確保 row_index 在範圍內
            if self.row_index < len(lines):
                new_data = lines[self.row_index]

                # 判斷新的資料
                if new_data[0] == '0':
                    self.update_traffic_light(0)  # 綠燈
                    self.alert_label.setText("Normal")
                elif new_data[0] == '1':
                    self.update_traffic_light(1)  # 紅燈
                    self.alert_label.setText("Warning")
                elif new_data[0] == '2':
                    self.update_traffic_light(1)  # 紅燈
                    self.alert_label.setText("Warning")
                else:
                    self.update_traffic_light(None)  # 灰燈
                    self.alert_label.setText("Waiting...")

                self.row_index += 1  # 增加索引以讀取下一行
            # else:
            #     # self.timer.stop()  # 停止計時器
            #     self.alert_label.setText("All data processed. Press Start to restart.")
            #     # self.start_button.setEnabled(True)  # 重新啟用開始按鈕
        except Exception as e:
            self.update_traffic_light(None)  # 灰燈
            self.alert_label.setText(f"Error: {str(e)}")

    def update_traffic_light(self, status):
        """根據狀態更新交通燈顏色"""
        if status == 0:
            self.traffic_light.setStyleSheet("border-radius: 25px; background-color: green;")
        elif status == 1:
            self.traffic_light.setStyleSheet("border-radius: 25px; background-color: red;")
        else:
            self.traffic_light.setStyleSheet("border-radius: 25px; background-color: gray;")

    def show_message(self, message):
        """顯示消息框，讓用戶點擊 OK 以繼續"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Information")
        msg_box.setText(message)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui_app = GuiApp()
    gui_app.show()

    sys.exit(app.exec_())
