from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch

def activate_sensor(YOLO_path = r"parameters\best_new.pt", 
                    cam_index = 1, # often 0 is the installed webcam in laptop
                    conf = 0.25, # recommend 0.9 or higher
                    inference_interval = 1.0, # in seconds
                    window_name = "Water Level Sensor",
                    A = 8,
                    B = 0,
                    height_deci = 1,
                    segmentation_view = True,
                    recording = False,
                    recording_name = "sensor_recording",
                    resolution_height = 480,
                    resolution_width = 640):
    
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

    try:
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

                # Box lable view
                if segmentation_view:
                    annotated = r0.plot()
                else:
                    annotated = frame.copy()

                # FPS calculation
                fps = 1.0 / (current_time - last_inference_time)

                # Time update
                last_inference_time = current_time

                # Image texting and output
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

                yield approx_h

                if recording:
                    out.write(annotated)

            # Close the window to break the detection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        print("Sensor shutting down safely...")
        if recording:
            out.release()
        cap.release()
        cv2.destroyAllWindows()