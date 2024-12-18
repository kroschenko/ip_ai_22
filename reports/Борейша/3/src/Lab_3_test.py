from ultralytics import YOLO
import cv2
import torch

model_path = "IP_Lab_3/blue_border/runs/yolov5n/weights/weights/best.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")
model = YOLO(model_path).to(device)

video_path = "Брест день.mp4"
output_path = "blue_border(day).mp4"

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for result in model.predict(source=video_path, stream=True, conf=0.5, device=device):
    frame = result.plot()
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()


print(f"Видео с детекциями сохранено в: {output_path}")