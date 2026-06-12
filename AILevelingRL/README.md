# PyArduino 개발
## Overview
* 아두이노 R4를 파이썬으로 제어하면 편리할듯
	* C언어 / 자바를 무시할 수 있다!
* 기존 PyFimata는 매우 구식!
```bash title=setup 
# 윈도우 환경에서 진행, git / conda 설치 등 필요함
# Arduino 포트확인하고 telemetrix 업로드해놓을 것
# 작업해보니 venv 에서도 잘 됨
# python -m venv venv
conda create -n pyarduino
conda activate pyarduino
conda install pip
pip install telemetrix-uno-r4
```
* 11/10 YOLO 업데이트 지원
```
conda create -n pyarduino2 python=3.10
pip install openvino ultralytics telemetrix-uno-r4 torch torchvision
```

* 01/13 MCP4728 (Pump) 업데이트 지원
```
기존 venv 폴더에 PyArduino, venv 파일 복붙할 것
```

## Code
```python title=simple_relay_controller
import sys
import time
from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi import telemetrix_uno_r4_wifi
import logging

class PyArduino():
    def __init__(self):
        self.board = telemetrix_uno_r4_wifi.TelemetrixUnoR4WiFi(transport_type=1)

    def run_digital_write(self, pin_number : int, pin_state : bool):
        self.board.set_pin_mode_digital_output(pin_number)

        if pin_state is True:
            self.board.digital_write(pin_number, 1)
            logging.info(f'{pin_number} is {pin_state}')

        elif pin_state is False:
            self.board.digital_write(pin_number, 0)
            logging.info(f'{pin_number} is {pin_state}')

        time.sleep(0.001)

        # self.board.shutdown()
        
    def get_analog_state(self, pin_number : int):
        global analog_value
        analog_value = None
        self.board.set_pin_mode_analog_input(pin_number, differential = 0, callback = self.get_state_slicer)

        while analog_value is None:
            time.sleep(0.01)

        # self.board.shutdown()

        return analog_value

    def get_state_slicer(self, data):
        global analog_value
        pin_mode_index = 0
        pin_number_index = 1
        pin_state_index = 2

        # print(data[pin_state_index])

        analog_value = data[pin_state_index]
    
def main():
    pa = PyArduino()
    value = pa.get_analog_state(2)
    print(value)

    for pin_number in range(4,8):
        pa.run_digital_write(pin_number, True)
        time.sleep(1)
        pa.run_digital_write(pin_number, False)

if __name__ is '__main__':
    main()

```
## Result
![[KakaoTalk_20251030_000552484.mp4]]




>>>>>>> 889aea2d9e4a65a27b8ad4d884df2dd088c5b590




