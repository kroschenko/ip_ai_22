import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prepare_data_with_augmentation():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader

def train_model(net, trainloader, epochs):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.8)

    loss_values = []

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                loss_values.append(running_loss / 100)
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), 'saved_weights.pth')
    print('Model saved!')
    return loss_values

def plot_error(loss_values):
    plt.plot(loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training')
    plt.show()

def test_model(net, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')


def first_model(net):
    trainloader, testloader = prepare_data_with_augmentation()

    loss_values = train_model(net, trainloader, 80)
    plot_error(loss_values)
    test_model(net, testloader)

def load_model(net):
    net.load_state_dict(torch.load('saved_weights.pth', weights_only=True))
    net.eval()


def visualize_image_and_prediction(model):
    app = QApplication(sys.argv)
    image_path, _ = QFileDialog().getOpenFileName(
        None, "Выберите изображение", "", "Image files (*.jpg *.jpeg *.png *.bmp *.webp)"
    )
    app.quit()

    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_label = classes[predicted_class.item()]

    plt.imshow(np.array(image))
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    net = SimpleCNN()

    # обучить заново
    # first_model(net)

    # загрузить модель
    load_model(net)

    visualize_image_and_prediction(net)