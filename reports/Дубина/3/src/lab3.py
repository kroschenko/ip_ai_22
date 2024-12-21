from ultralytics import YOLO

# Создание модели (или загрузка предобученной)
model = YOLO('yolov5s.pt')

if __name__ == '__main__':
    model.train(data='data.yaml', epochs=200, batch=32, imgsz=640, project='runs/train', name='yolo_train_exp')
