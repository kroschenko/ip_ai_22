import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f'#{epoch + 1:2d}', end=' ')
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Loss: {running_loss / len(train_loader):.4f}')

def test_model(model, test_loader):
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
    print(f'Accuracy: {100 * correct / total:.2f}%')

def test_single_image(model, image_path):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)

    image = image.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.Softmax(dim=1)(output)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    print(f'Predicted Class: {classes[predicted_class]}')
    print(f'Probabilities: {probabilities.cpu().numpy()}')


def main():
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier[6] = nn.Linear(4096, 10)
    alexnet = alexnet.to(device)

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    for param in alexnet.parameters():
        param.requires_grad = False

    for param in alexnet.classifier[6].parameters():
        param.requires_grad = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9)

    train_model(alexnet, train_loader, criterion, optimizer)
    torch.save(alexnet.state_dict(), './w.pth')
    test_model(alexnet, test_loader)

    image_path = '0.png'
    test_single_image(alexnet, image_path)


if __name__=="__main__":
    main()
