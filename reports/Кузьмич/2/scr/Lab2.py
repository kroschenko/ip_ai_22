import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Настройка преобразования данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Функция для загрузки и подготовки данных
def load_data():
    try:
        print("Загружаем данные MNIST...")
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        print("Данные загружены успешно.")
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        print("DataLoaders созданы.")
        return train_loader, test_loader
    except Exception as e:
        print("Ошибка при загрузке данных:", e)

# Определение архитектуры модели
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Функция обучения модели
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    try:
        train_losses, test_losses = [], []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Тестирование модели
            model.eval()
            test_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            accuracy = correct / len(test_loader.dataset)

            print(f"Эпоха {epoch+1}/{epochs}, Ошибка на обучении: {train_loss:.4f}, Ошибка на тесте: {test_loss:.4f}, Точность на тесте: {accuracy*100:.2f}%")
        
        return train_losses, test_losses
    except Exception as e:
        print("Ошибка в процессе обучения:", e)

# Основной блок выполнения
try:
    train_loader, test_loader = load_data()
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print("Начало обучения...")
    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10)
    
    # Построение графика ошибки
    plt.plot(train_losses, label='Ошибка на обучении')
    plt.plot(test_losses, label='Ошибка на тесте')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.title('График ошибки на обучающей и тестовой выборках')
    plt.show()

    print("Сравнение: Современные модели для MNIST, такие как ResNet и EfficientNet, достигают точности выше 99%.")
    print("Наша простая CNN показывает результат, достаточный для базовых задач, но уступает в точности более сложным моделям.")

    # Функция для визуализации предсказания модели
    def visualize_prediction(model, dataset, index=0):
        model.eval()
        image, label = dataset[index]
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f"Реальный класс: {label}, Предсказание: {predicted.item()}")
        plt.show()

    print("Визуализация предсказания модели...")
    visualize_prediction(model, test_loader.dataset, index=7)

except Exception as e:
    print("Произошла ошибка:", e)
