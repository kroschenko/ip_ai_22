import os
import cv2
from ultralytics import YOLO

# Пути к модели и данным
model_path = "D:/7 семестр/ОИвИС лабы/ОИвИС лаба №3/yolov8n.pt"
data_yaml_path = "D:/7 семестр/ОИвИС лабы/ОИвИС лаба №3/Для фотографий main_road/yolo_data_for_main_road/data.yaml"
input_video_path = "D:/7 семестр/ОИвИС лабы/ОИвИС лаба №3/Брест ночь.mp4"
output_video_path = "D:/7 семестр/ОИвИС лабы/ОИвИС лаба №3/Brest_night первоначально не трогать!!!!.mp4"

model = YOLO(model_path)

model.train(data=data_yaml_path, epochs=50, imgsz=360)

cap = cv2.VideoCapture(input_video_path)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Original resolution: {original_width}x{original_height}, FPS: {fps}, Total frames: {frame_count}")

scale_factor = 0.5
new_width = int(original_width * scale_factor)
new_height = int(original_height * scale_factor)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, 10, (new_width, new_height))

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    results = model(resized_frame, conf=0.7, iou=0.5, max_det=100)

    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    frame_index += 1
    print(f"Processed frame {frame_index}/{frame_count}")

cap.release()
out.release()
print("\nProcessing completed. Video saved as:", output_video_path)