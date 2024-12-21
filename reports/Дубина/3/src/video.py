from ultralytics import YOLO

model = YOLO('D:/PyCharm project/ОИВИС/3/runs/train/yolo_train_exp9/weights/best.pt')

results = model.predict('D:/PyCharm project/ОИВИС/3/Брест ночь.mp4', show=True, save=True)