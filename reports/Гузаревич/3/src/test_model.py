from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict('../1.jpg', show=True, save=True)
