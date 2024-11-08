import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 256 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_data = STL10(root='./data', split='train', download=True, transform=transform)

train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 20
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
early_stop_counter = 0
early_stop_patience = 5
best_val_loss = np.inf

if os.path.exists("model.pth"):
    print("Загрузка модели из model.pth")
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
else:
    print("Модель не найдена. Начинаю обучение.")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        scheduler.step(val_loss)

        print(
            f"Эпоха {epoch + 1}/{num_epochs}, Потери (тренировочные): {train_loss:.4f}, Точность (тренировочные): {train_accuracy:.2f}%, Потери (валидационные): {val_loss:.4f}, Точность (валидационные): {val_accuracy:.2f}%")
        print(f"Текущая скорость обучения: {scheduler.optimizer.param_groups[0]['lr']}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "model.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print("Ранняя остановка обучения.")
            break

    print("веса модели сохранены в 'model.pth'.")

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Тренировочные потери')
plt.plot(epochs, val_losses, label='Валидационные потери')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('График изменения потерь')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Тренировочная точность')
plt.plot(epochs, val_accuracies, label='Валидационная точность')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.title('График изменения точности')
plt.legend()

plt.tight_layout()
plt.show()

# Классы STL-10
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((96, 96))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100
    return classes[predicted_class], confidence

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return

    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    predicted_class, confidence = predict_image(file_path)
    result_label.config(text=f"Предсказание: {predicted_class}\nУверенность: {confidence:.2f}%")

root = tk.Tk()
root.title("Распознавание изображений с помощью CNN")

load_button = tk.Button(root, text="Загрузить изображение", command=load_image)
load_button.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Загрузите изображение для распознавания", font=("Arial", 14))
result_label.pack()

root.mainloop()
