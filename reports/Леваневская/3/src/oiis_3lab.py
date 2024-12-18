from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data='/content/sign/data.yaml',
    epochs=10,
    imgsz=1280,
    batch=8,
    verbose=True,
    plots=True,
    project='/content/drive/MyDrive/sign',
    name='sign',
)
