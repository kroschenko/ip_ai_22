from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='/content/sign/data.yaml',
    epochs=10,
    imgsz=1280,
    batch=4,
    verbose=True,
    plots=True,
    project='/content/drive/MyDrive/sign',
    name='sign',
    # resume=True
    # save_dir='/content/drive/MyDrive'
)