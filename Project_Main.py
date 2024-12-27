import subprocess
import time

def run_a_and_b():
    # 啟動 a.py
    process_a = subprocess.Popen(["python", "model.py"])

    # 等待一小段時間確保 a.py 已經開始執行
    time.sleep(2)

    # 啟動 b.py
    process_b = subprocess.Popen(["python", "UI.py"])

    # # 等待兩個程式結束執行
    process_a.wait()
    process_b.wait()

if __name__ == "__main__":
    run_a_and_b()
