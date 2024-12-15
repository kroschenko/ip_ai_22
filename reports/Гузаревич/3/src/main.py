import os
import shutil
import pandas as pd
from PIL import Image

csv_path = "F:/Univercity/4kurs/OIVIS/Completed/OIVIS_Lab_3/rtsd-d3-gt/blue_rect/test_gt.csv"
images_folder = "F:/Univercity/4kurs/OIVIS/Completed/OIVIS_Lab_3/rtsd-d3-frames/test"
output_folder = "F:/Univercity/4kurs/OIVIS/Completed/OIVIS_Lab_3/try"

yolo_labels_folder = os.path.join(output_folder, "labels")
yolo_images_folder = os.path.join(output_folder, "images")
classes_file = os.path.join(output_folder, "classes.txt")

os.makedirs(yolo_labels_folder, exist_ok=True)
os.makedirs(yolo_images_folder, exist_ok=True)

data = pd.read_csv(csv_path)

unique_classes = sorted(data['sign_class'].unique())
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

with open(classes_file, "w") as f:
    for class_name, class_index in class_to_index.items():
        f.write(f"{class_name}\n")

def convert_to_yolo(image_width, image_height, x, y, width, height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return x_center, y_center, norm_width, norm_height

for _, row in data.iterrows():
    filename = row['filename']
    x_from = row['x_from']
    y_from = row['y_from']
    width = row['width']
    height = row['height']
    sign_class = row['sign_class']

    class_index = class_to_index[sign_class]

    image_path = os.path.join(images_folder, filename)

    if not os.path.exists(image_path):
        print(f"Изображение {filename} не найдено.")
        continue

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    x_center, y_center, norm_width, norm_height = convert_to_yolo(
        img_width, img_height, x_from, y_from, width, height
    )

    yolo_label_path = os.path.join(
        yolo_labels_folder, os.path.splitext(filename)[0] + ".txt"
    )
    with open(yolo_label_path, "a") as label_file:
        label_file.write(f"{class_index} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    shutil.copy(image_path, yolo_images_folder)

print("Преобразование завершено")