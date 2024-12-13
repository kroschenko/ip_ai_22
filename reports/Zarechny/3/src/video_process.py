import torch
import cv2

# Путь к видео и весам модели
video_path = '/home/azarechny/university/OIvIS/3/src/video/Brest_day.mp4'  # Путь к входному видео
weights_path = '/home/azarechny/university/OIvIS/3/src/runs/train/exp5/weights/best.pt'  # Путь к вашим обученным весам

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Открытие видео
cap = cv2.VideoCapture(video_path)

# Получение параметров видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Создание объекта для записи обработанного видео (опционально)
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Обработка кадров
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Выполнение детекции
    results = model(frame)

    # Получение предсказанного изображения с разметкой
    annotated_frame = results.render()[0]

    # Отображение детекций в реальном времени (опционально)
    cv2.imshow('YOLOv5 Detection', annotated_frame)

    # Запись обработанного кадра в выходное видео
    out.write(annotated_frame)

    # Нажмите 'q' для выхода из отображения
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
