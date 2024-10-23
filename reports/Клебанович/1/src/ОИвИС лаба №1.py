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
from PIL import Image, ImageOps, ImageTk
import numpy as np

batch_size = 64
learning_rate = 0.001
num_epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
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

        image_tensor = transform_pipeline(inverted_image).unsqueeze(0)
        
        with torch.no_grad():
            model.eval()
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        class_name = classes[predicted.item()]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(image, cmap='gray', interpolation='none')
        axes[0].set_title("Исходное изображение")
        axes[0].axis('off')

        axes[1].imshow(image_tensor[0][0], cmap='gray', interpolation='none')
        axes[1].set_title(f"Измененное изображение\nПрогноз: {class_name}")
        axes[1].axis('off')

        plt.show()

root = tk.Tk()
root.title("Классификация изображений")

open_button = tk.Button(root, text="Выберите фотографию", command=open_image)
open_button.pack()

root.mainloop()