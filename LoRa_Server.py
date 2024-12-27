import serial
from torch_vggish_yamnet import yamnet
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch_vggish_yamnet import yamnet
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
#from openpyxl import Workbook, load_workbook
import csv

      
class YAMNetClassifier(nn.Module):
    def __init__(self, num_classes=27):
        super(YAMNetClassifier, self).__init__()
        self.yamnet = yamnet.yamnet(pretrained=True)
        for param in self.yamnet.parameters():
            param.requires_grad = False  # 冻结预训练模型参数
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 自定义分类层
    def forward(self, x):
        x = self.yamnet(x)
        print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

 #類別分類



# 設定串列埠參數 (根據你的 USB-to-LoRa 模組修改 port)
ser = serial.Serial(port='COM4', baudrate=9600, timeout=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YAMNetClassifier(num_classes=27).to(device)
model_path = "yamnet_classifier_100.pth"  # 替換成你的 .pth 文件路徑
model.load_state_dict(torch.load(model_path, map_location=device))

print("Listening for LoRa messages from Arduino...")

# 每次接收到的資料儲存為整數陣列
model.eval()

while True:
    # 讀取串列埠接收到的資料
    data = ""
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8', errors='ignore').strip()  # 解碼並去除多餘的空格或換行
        # print(f"Received part: {data}")
        
        # 檢查資料是否為數字格式（如果是數字並且以逗號隔開）
        # print(len(data))
        if len(data) != 0:
            # print(data)
        # if all(c.isdigit() or c == ',' for c in data):  # 確保資料中只包含數字或逗號
            #將接收到的資料轉換為 int 陣列
            data_array = [int(x) for x in data.split(',') if x.strip().isdigit()]  # 去除空格並確保每個元素是數字
            # 印出資料陣列
            print(f"Array: {data_array}")
            # print(type(data_array))


            
            # 進行資料處理
            # print(f"Data processed: {data_array}")
            while len(data_array) < 64:
                data_array.append(0)
            new_list = []
            for i in data_array:
                n = (i * -1 )/ 10 
                new_list.append(n)
            mel_mean = np.array(new_list)
            mel_tensor = torch.tensor(mel_mean, dtype=torch.float32)
            # print(f"mel_tensor shape before unsqueeze: {mel_tensor.shape}")
            #print(mel_tensor)
            print(new_list)
            ser.close()
            break
        else:
            print("Received non-numeric data, skipping...")


  
        # with torch.no_grad():
            
        #     # inputs = mel_tensor.to(device)
        #     inputs = mel_tensor.unsqueeze(0).to(device)
        #         # inputs = inputs.unsqueeze(1)  # YAMNet 要求输入形状为 (batch, 1, num_samples)
        #         # inputs = inputs.squeeze(1)
        #     outputs = model(inputs)
        #     # print(outputs.shape)
        #     _, predicted = torch.max(outputs, dim=1)
        #     # pred_list.append(predicted)
        #     print(predicted.item())
            
        # output = str(predicted.item())
        # print(type(output))
        # output_to_excel= classify(output)
        # # 初始化 Excel 文件
        # file_name = "class_detection.csv"

        # # 定義寫入函數
        # def write_to_csv(file_name, value):
        #     # 追加數據到 CSV
        #     with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        #         writer = csv.writer(file)
        #         writer.writerow([value])
        #     print(f"成功將數據 {value} 寫入 {file_name}！")

        # # 模型推理結果（替換為 classify(output) 的返回值）
        # output_to_excel = 2  # 假設 classify(output) 的返回值為 1

        # # 每次執行時寫入 CSV
        # write_to_csv(file_name, output_to_excel)