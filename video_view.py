import os
import cv2
from ultralytics import YOLO

def process_video(
        video_path:str,
        size:int = 640,
        show:bool = True,
        save:bool = False,
    ) -> str:
    # Load the YOLOv8 model
    model_path = os.path.join('waste_detection', 'yolo', 'weights', 'best.pt')
    model = YOLO(model_path)
    results = model(video_path, save=save, show=show, imgsz=size)
    dir = results[0].save_dir
    del results
    return dir

if __name__ == '__main__':
    video_path = os.path.join('video', 'waste.mp4')
    print(process_video(video_path))