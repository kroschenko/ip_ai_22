import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 24 * 24, 512)  # Подходящие размеры после сверток и пуллинга
        self.fc2 = nn.Linear(512, 10)  # STL-10 имеет 10 классов

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 24 * 24)  # Разворачиваем тензор перед полносвязным слоем
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Определение архитектуры сверточной нейронной сети


    # Гиперпараметры
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10
    model_save_path = 'simple_cnn_stl10.pth'  # Путь для сохранения модели

    # Преобразования для увеличения данных и нормализации
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Нормализация с средним 0.5 и стандартным отклонением 0.5
    ])

    # Загрузка датасета STL-10
    train_dataset = STL10(root='./data', split='train', transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = STL10(root='./data', split='test', transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Инициализация модели, функции потерь и оптимизатора
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print("Модель загружена из файла.")

    else:
        # Тренировочный цикл
        pass
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Переместить данные на устройство (CPU или GPU)
            inputs, labels = inputs, labels

            # Обнулить градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            # Статистика потерь
            running_loss += loss.item()
            if i == 0:  # Печатаем каждые 100 мини-батчей
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.2e}')
                running_loss = 0.0

    torch.save(model.state_dict(), model_save_path)
    print("Модель сохранена в файл.")

    # Оценка точности модели на тестовой выборке
    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0
    with torch.no_grad():  # Отключаем вычисление градиентов для оценки
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Точность модели на тестовой выборке: {accuracy:.2f}%')


if __name__ == "__main__":
    main()
