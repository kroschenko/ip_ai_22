import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import SqueezeNet1_1_Weights
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label
import torch.nn.functional as F
import os

MODEL_PATH = "model.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

# 2. Загрузка и адаптация модели
model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1, 1), stride=(1, 1))  # 100 классов CIFAR-100
model.num_classes = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Модель загружена из сохраненного файла.")
else:
    print("Обучение модели...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    num_epochs = 10
    train_losses = []
    test_losses = []
    test_accuracies = []  # Список для хранения точности на тестовой выборке

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Тестирование
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Считаем количество верных предсказаний
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        # Вычисляем точность
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} completed. Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена в файл {MODEL_PATH}.")

    # Построение графиков потерь и точности
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Over Epochs')
    plt.legend()

    plt.show()


cifar100_classes = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


def predict_image(img_path):
    model.eval()

    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        top_prob, top_class = torch.topk(probabilities, 1, dim=1)

    predicted_class = top_class.item()
    confidence = top_prob.item()
    class_name = cifar100_classes[predicted_class]
    return predicted_class, class_name, confidence


def open_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((224, 224))
        img_display = ImageTk.PhotoImage(img)
        panel.config(image=img_display)
        panel.image = img_display

        pred_class, class_name, confidence = predict_image(file_path)

        result_text = f"Предсказанный класс: {pred_class} ({class_name})\nУверенность: {confidence:.2f}"
        result_label.config(text=result_text)


root = tk.Tk()
root.title("Классификатор изображений")
root.geometry("400x400")

message = tk.Label(root, text="Нажмите 'Выбрать изображение' для загрузки и классификации", pady=10)
message.pack()

button = tk.Button(root, text="Выбрать изображение", command=open_and_predict)
button.pack()

panel = Label(root)
panel.pack()

result_label = tk.Label(root, text="Результат предсказания", pady=10)
result_label.pack()

root.mainloop()
