from torch_vggish_yamnet import yamnet
import multiprocessing
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
#from openpyxl import Workbook, load_workbook
import csv
# from Lora_server import data_array
import time
import importlib
import warnings
warnings.filterwarnings('ignore')
# from Lora_server import data_array
# 你的資料字串
# data_str = "51,101,117,149,150,173,210,211,206,235,254,268,283,273,272,283,289,304,313,308,325,324,311,324,336,332,325,335,330,326,332,338,336,342,343,360,367,360,359,375,386,385,383,394,409,422,425,440,445,441,455,460,460,463,464,462,464,465,471,478,505,551,648,764"

# # 將字串拆分並轉換為數字列表
# data_list = [float(x) for x in data_str.split(',')]
# temp = "021"
while True:
    import Lora_server
    importlib.reload(Lora_server)
    # from Lora_server import data_array

    data_list = Lora_server.data_array

    while len(data_list) < 64:
        data_list.append(0)
    new_list = []
    for i in data_list:
        n = (i * -1 )/ 10 
        new_list.append(n)
    # print(new_list)
    # 轉換為 NumPy 陣列
    mel_mean = np.array(new_list)
    mel_tensor = torch.tensor(mel_mean, dtype=torch.float32)
    # print(f"mel_tensor shape before unsqueeze: {mel_tensor.shape}")
    # print(mel_tensor)
        
    class YAMNetClassifier(nn.Module):
        def __init__(self, num_classes=27):
            super(YAMNetClassifier, self).__init__()
            self.yamnet = yamnet.yamnet(pretrained=True)
            for param in self.yamnet.parameters():
                param.requires_grad = False  # 冻结预训练模型参数
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, num_classes)  # 自定义分类层
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    #類別分類
    def classify(value):
        # 電鋸、斧頭等
        Machine_and_Tools = {'1', '7', '10', '11', '12', '13', '14', '15', '16'}
        # 腳步聲、說話聲等
        Human = {'9', '20', '17', '18', '19'}
        # 正常類別 (normal)
        normal = {'2', '3', '4', '5', '6', '8', '21', '22', '23', '24', '25', '26', '27'}

        if value in Machine_and_Tools:
            return 1
        elif value in Human:
            return 2
        elif value in normal:
            return 0
        else:
            return None, 'unknown'  # 如果值不在任何分類中



    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YAMNetClassifier(num_classes=27).to(device)
    model_path = "yamnet_classifier_100.pth"  # 替換成你的 .pth 文件路徑
    model.load_state_dict(torch.load(model_path, map_location=device))


    pred_list = []

        
    with torch.no_grad():
        
        inputs = mel_tensor.to(device)
            # inputs = inputs.unsqueeze(1)  # YAMNet 要求输入形状为 (batch, 1, num_samples)
            # inputs = inputs.squeeze(1)
        outputs = model(inputs)
        # print(outputs.shape)
        _, predicted = torch.max(outputs, dim=0)
        pred_list.append(predicted)
        
        
    output = str(predicted.item())
    # print(type(output))
    output_to_excel= classify(output)
    # 初始化 Excel 文件
    file_name = "class_detection.csv"
    

    # 定義寫入函數
    def write_to_csv(file_name, value):
        # 追加數據到 CSV
        with open(file_name, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([value])
        print("prediction:", value)
        print(f"成功將數據 {value} 寫入 {file_name}！")

    # 模型推理結果（替換為 classify(output) 的返回值）
    # output_to_excel = 2  # 假設 classify(output) 的返回值為 1

    # 每次執行時寫入 CSV
    write_to_csv(file_name, output_to_excel)
    time.sleep(20)