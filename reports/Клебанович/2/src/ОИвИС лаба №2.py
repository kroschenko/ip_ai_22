import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Параметры обучения
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# Определяем устройство (GPU, если доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Преобразования для данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загрузка данных Fashion-MNIST
train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Используем предобученную сеть ResNet34 и изменим архитектуру для Fashion-MNIST
class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        # Загрузим предобученную ResNet34 и заменим первый слой под один канал (для серых изображений)
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Заменим последний fully connected слой на слой с 10 выходами (для классов Fashion-MNIST)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)
        # Заморозим начальные слои
        for param in self.model.parameters():
            param.requires_grad = False
        # Разморозим только последний полностью связанный слой
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

model = ModifiedResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

scaler = GradScaler()

def train_model():
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_loss

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

train_loss = train_model()
evaluate_model()

plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

classes = ['Футболка/топ', 'Брюки', 'Свитер', 'Платье', 'Пальто', 
           'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботинки']

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        inverted_image = ImageOps.invert(image.convert('RGB'))

        transform_pipeline = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image_tensor = transform_pipeline(inverted_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            model.eval()
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        class_name = classes[predicted.item()]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image, cmap='gray', interpolation='none')
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')

        axes[1].imshow(image_tensor.cpu().numpy()[0][0], cmap='gray', interpolation='none')
        axes[1].set_title(f"Измененное изображение\nПрогноз: {class_name}")
        axes[1].axis('off')

        plt.show()

root = tk.Tk()
root.title("Классификация изображений")

open_button = tk.Button(root, text="Выберите фотографию", command=open_image)
open_button.pack()

root.mainloop()