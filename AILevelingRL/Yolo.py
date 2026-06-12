from ultralytics import YOLO
import time

# conda create -n pyarduino2 python=3.10
# pip install openvino ultralytics telemetrix-uno-r4 torch torchvision

class YoloMode():
    def __init__(self):
        # self.model = YOLO("yolo11n.pt")
        self.model = YOLO("yolo12m.pt")
        # self.model = YOLO("yolov13n.pt")
    
    def run_trainer(self):
        nano_batch = 100
        nano_array = [1,2]

        medium_batch = 40
        medium_array = [0,1,2]

        results = self.model.train(data="/workspace/sungwoo_docker/AILevelingRL/DatasetMK2/data.yaml", epochs=300, imgsz=800, batch=medium_batch ,device=medium_array)

    def run_openvino_exporter(self, path):
        self.model = YOLO(path)
        self.model.export(format="openvino")

    def run_eval(self, path):
        self.model = YOLO(path)
        source = "C:\\Users\\USER_55_DESKTOP\\Desktop\\workspace\\AILevelingRL\\models\\Dataset0\\valid\\images\\251104_v1_1_mp4-0071_jpg.rf.f78dfd96c7bf6e821948e25855280ff1.jpg"
        results = self.model.predict(source, imgsz=800)
    
    def run_openvino_eval(self, path):
        self.model = YOLO(path)
        source = "C:\\Users\\USER_55_DESKTOP\\Desktop\\workspace\\AILevelingRL\\models\\Dataset0\\valid\\images\\251104_v1_1_mp4-0071_jpg.rf.f78dfd96c7bf6e821948e25855280ff1.jpg"
        results = self.model(source, device="intel:gpu")

def main():
    ym = YoloMode()
    ym.run_trainer()
    # model_path = "C:\\Users\\USER_55_DESKTOP\\Desktop\\workspace\\AILevelingRL\\models\\13n.pt"
    # model_path = 'C:\\Users\\USER_55_DESKTOP\\Desktop\\workspace\\AILevelingRL\\models\\12n_openvino_model'
    # ym.run_eval(model_path)
    # ym.run_openvino_exporter(model_path)
    # ym.run_openvino_eval(model_path)

if __name__ == "__main__":
    main()
