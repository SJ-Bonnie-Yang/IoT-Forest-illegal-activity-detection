from SX127x.LoRa import LoRa
from SX127x.board_config import BOARD

BOARD.setup()

class LoRaSender(LoRa):
    def __init__(self):
        super(LoRaSender, self).__init__(verbose=False)

    def send_message(self, message):
        self.write_payload([ord(c) for c in message])
        self.set_mode_tx()
        print(f"Message sent: {message}")

if __name__ == "__main__":
    lora = LoRaSender()
    try:
        lora.set_mode(MODE.SLEEP)
        lora.set_pa_config(pa_select=1)  # 設置功率
        lora.send_message("Hello LoRa!")
    finally:
        BOARD.teardown()