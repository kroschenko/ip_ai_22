import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = 'model_weights.pth'  # Путь для сохранения весов модели

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Инициализация модели
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)  # Изменение последнего слоя для 10 классов (цифры)
model = model.to(device)

# Определение оптимизатора и функции потерь
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    loss_history = []  # Список для сохранения значений потерь

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)  # Сохранение потери после каждой эпохи
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Сохранение весов после обучения
    torch.save(model.state_dict(), weights_path)

    # Построение графика ошибок
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.show()


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Точность: {100 * correct / total:.2f}%')


def test_image(model):
    model.eval()

    app = QApplication(sys.argv)
    image_path, _ = QFileDialog().getOpenFileName(
        None, "Выберите изображение", "", "Image files (*.jpg *.jpeg *.png *.bmp *.webp)"
    )
    app.quit()

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    predicted_label = predicted.item()

    plt.imshow(image, cmap='gray')
    plt.title(f"Распознано: {predicted_label}")
    plt.axis('off')
    plt.show()


# Проверка наличия сохранённых весов
if os.path.exists(weights_path):
    # Загрузка весов, если файл существует
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Загружены сохранённые веса модели.")
else:
    # Обучение модели и сохранение весов
    print("Файл с весами не найден. Начинаем обучение...")
    num_epochs = 10
    train(model, train_loader, criterion, optimizer, num_epochs)
    print("Обучение завершено. Веса сохранены.")

# Тестирование модели
test(model, test_loader)

# Тестируем модель на изображении
test_image(model)