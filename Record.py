"""將麥克風連接到RPi
=>在RPi裡面執行此code來錄製real time 音訊
=>音訊會經過Mel 平埔處理成一串數字後
=>輸入到訓練好的模型做辨識
"""

import pyaudio
import wave
import os

def record_audio_to_file(duration, folder, filename, sample_rate=16000, channels=1, chunk_size=1024):
    """
    使用 pyaudio 錄製音訊並將其保存為 WAV 檔案到指定資料夾。
    :param duration: 錄音時間（秒）
    :param folder: 儲存音訊檔案的資料夾
    :param filename: 音訊檔案名稱
    :param sample_rate: 音訊採樣率（預設 16000 Hz）
    :param channels: 聲道數量（預設單聲道）
    :param chunk_size: 單次讀取的音訊塊大小
    """
    # 確保資料夾存在
    os.makedirs(folder, exist_ok=True)
    
    # 完整的檔案路徑
    filepath = os.path.join(folder, filename)

    # 初始化 PyAudio 物件
    p = pyaudio.PyAudio()

    # 開啟音訊流
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print(f"開始錄製 {duration} 秒音訊，將保存為 {filepath}...")

    frames = []

    # 錄製音訊
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    # 停止並關閉音訊流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 儲存錄音為 WAV 檔案
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"音訊已保存到 {filepath}")

# 使用範例
if __name__ == "__main__":
    duration = 10  # 錄製 10 秒鐘的音訊
    folder = "Audio"  # 音訊檔案儲存的資料夾
    filename = "test_audio.wav"  # 儲存的音訊檔案名稱
    record_audio_to_file(duration, folder, filename)
