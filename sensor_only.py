from ultralytics import YOLO
import cv2
import time
import pickle
import numpy as np
from pathlib import Path

from .AILevelingRL import PyArduino

with open(r"parameters\linear_regression_param.pickle", "rb") as f:
        param = pickle.load(f)
A, B = param["grad"], param["int"]

pa = PyArduino("minima")

time_list = []
cv_list = []

def activate_sensor(YOLO_path = r"data\final_result\best.pt", 
                    cam_index = 1, # often 0 is the installed webcam in laptop
                    conf = 0.25, # recommend 0.9 or higher
                    inference_interval = 1.0, # probably in second? I guess?
                    window_name = "Water Level Sensor",
                    A = A,
                    B = B,
                    height_deci = 1,
                    segmentation_view = True,
                    recording = False,
                    recording_name = "sensor_recording",
                    resolution_height = 480,
                    resolution_width = 640,
                    pump_speed = 14,
                    upper_valve = 5,
                    lower_valve = 6):
    
    model = YOLO(YOLO_path)

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}")
    
    # default webcam resolution at 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)

    # dafult recoding is False
    if recording:
        fps_out = 1.0/inference_interval
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(f"{recording_name}.mp4", fourcc, fps_out, (resolution_width, resolution_height))

    # initialising variables for limiting inference time
    last_inference_time = 0
    approx_h = None

    print(f"Inference started.")

    pa.run_digital_write(upper_valve, True)
    pa.run_digital_write(lower_valve, True)
    pa.run_pump_speed(pump_speed)

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"frame not captured.")
            break

        current_time = time.time()

        if current_time - last_inference_time >= inference_interval:
            # OBB inference
            result = model.predict(source=frame, conf=conf, verbose=False)
            r0 = result[0]

            # checking if tank and water is successfully detected by YOLO
            name_to_id = {v: k for k, v in r0.names.items()}
            tank_id = name_to_id.get("Tank")
            water_id = name_to_id.get("Water")
            detected_water = (r0.obb.cls.int() == water_id).any().item()
            detected_tank  = (r0.obb.cls.int() == tank_id).any().item()
            all_tank_rows = np.where(r0.obb.cls == tank_id)[0]
            all_water_rows = np.where(r0.obb.cls == water_id)[0]

            # there is a few cases where the object that is not a tank detected as a tank like a whiteboard with black rim.
            # these cases are handled via selecting the the true tank or water using the confidence value.
            tank_conf = r0.obb.conf[all_tank_rows]
            water_conf = r0.obb.conf[all_water_rows]

            # Water level inference
            # there is a few cases where the object that is not a tank detected as a tank like a whiteboard with black rim.
            # recommend to remove such case via setting high confidence value.
            if detected_tank:
                tank_row = all_tank_rows[np.argmax(tank_conf)]
                tank_h = r0.obb.xywhr[tank_row,3] # OBB.xywhr data is in a shape of (x-center, y-center, width, height, rotation in radian)
                if detected_water:
                    water_row = all_water_rows[np.argmax(water_conf)]
                    water_h = r0.obb.xywhr[water_row,3]
                    approx_h = (water_h/tank_h)*A + B
                    approx_h = approx_h.item()
                else:
                    approx_h = 0 # in case water is not detected, assume that the tank is empty
            else:
                approx_h = None # in case tank is not detected, we assume that the YOLO model cannot detect the system, thus cannot work as a sensor

            cv_list.append(approx_h)
            time_list.append(current_time)

            # box lable view
            if segmentation_view:
                annotated = r0.plot()
            else:
                annotated = frame.copy()

            # fps calculation
            fps = 1.0 / (current_time - last_inference_time)

            # time update
            last_inference_time = current_time

            # image texting and output
            if approx_h is not None:
                cv2.putText(
                    annotated,
                    f"Height: {np.round(approx_h, decimals=height_deci)}cm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2
                )
            else:
                cv2.putText(
                    annotated,
                    f"Height: Could NOT Detect Tank",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2
                )
            
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2
            )

            cv2.putText(
                annotated,
                f"cam index: {cam_index}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 0),
                thickness=1
            )

            cv2.imshow(window_name, annotated)

            if recording:
                out.write(annotated)

        # close the window to break the detection
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            pa.run_pump_speed(0)
            break

    if recording:
        out.release()

    cap.release()
    cv2.destroyAllWindows()

    return cv_list, time_list


def next_available_path(base_path: str) -> Path:
    """
    base_path가 존재하면 _01, _02 ...를 붙여서 비어있는 경로를 반환.
    예) pump_speed14.pickle -> pump_speed14_01.pickle -> ...
    """
    p = Path(base_path)

    # base 파일이 없으면 그대로 사용
    if not p.exists():
        return p

    stem = p.stem          # "pump_speed14"
    suffix = p.suffix      # ".pickle"
    parent = p.parent

    # _01 ~ _99 ... 계속 증가
    i = 1
    while True:
        candidate = parent / f"{stem}_{i:02d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


if __name__ == "__main__":
    pump_speed = 14
    cv_list, time_list = activate_sensor(YOLO_path = r"best_new.pt", 
                                         cam_index = 0, 
                                         conf = 0.90,
                                         inference_interval = 1,
                                         segmentation_view = False,
                                         recording = False,
                                         pump_speed = pump_speed
                                         )
    
    result = {"pump_speed": pump_speed,
              "cv": cv_list,
              "time": time_list}
    
    base = rf"data\steady_state_map\pump_speed{pump_speed}.pickle"
    save_path = next_available_path(base)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to: {save_path}")
