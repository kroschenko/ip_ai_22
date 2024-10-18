from PIL import Image
import torchvision.transforms as transforms
import torch
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from  main import SimpleCNN
from matplotlib import pyplot as plt

model = SimpleCNN()
model_save_path = 'simple_cnn_stl10.pth'  # Путь для сохранения модели

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))

# Путь к изображению
image_path = 'asset.png'

# Преобразования для изображения
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # Изменяем размер изображения на 96x96 (размер для STL-10)
    transforms.ToTensor(),  # Преобразуем изображение в тензор
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализуем
])

# Загрузка изображения и применение преобразований
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)  # Добавляем размер для batch

# Модель должна быть в режиме оценки
model.eval()

# Предсказание модели
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

# Маппинг предсказанного индекса к классу STL-10 (0-9)
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
predicted_class = classes[predicted.item()]

# Визуализация изображения и предсказания
plt.imshow(image)
plt.title(f'Предсказанный класс: {predicted_class}')
plt.axis('off')
plt.show()
