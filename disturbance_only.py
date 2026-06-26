from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch
import os
import pickle

from AILevelingRL.PyArduino import PyArduino
from sensor_control import activate_sensor


start_time = time.perf_counter()

if __name__ == "__main__":
    pa = PyArduino("minima")

    pa.run_digital_write(5, True)
    pa.run_digital_write(6, True)
    pa.run_digital_write(7, True)

    pa.run_pump_speed(0.0)

    sensor = activate_sensor(YOLO_path=r"parameters\YOLO.pt", A=7.701273019117101, B=0.7084194867553455)

    cv_list = []

    try:
        for cv in sensor:
            cv_list.append(cv)

            if time.perf_counter() - start_time >= 15 * 60:
                break

    finally:
        sensor.close()

    os.makedirs(r"data\control_result\disturbance_only", exist_ok=True)
    for i in range(0, 1000):
        filename = rf"data\control_result\disturbance_only\lower_tank_{i}.pickle"
        if not os.path.exists(filename):
            break

    with open(filename, "wb") as f:
        pickle.dump(cv_list, f)

    pa.run_digital_write(7, False)