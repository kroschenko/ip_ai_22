import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
image_path = '5.png'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('dataset/', train=True, download=False, transform=transform),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('dataset/', train=False, download=False, transform=transform),
    batch_size=batch_size_test, shuffle=False)

train_losses = []
train_counter = []


def train(epoch):
    alexnet.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = alexnet(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_counter.append((batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset)))
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    torch.save(alexnet.state_dict(), 'alexnet_model.pth')
    torch.save(optimizer.state_dict(), 'alexnet_optimizer.pth')


def test():
    alexnet.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = alexnet(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')


def plot_loss():
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples seen')
    plt.ylabel('CrossEntropyLoss')
    plt.title('График падения ошибки во время обучения')
    plt.show()


criterion = nn.CrossEntropyLoss()
optimizer = None

if os.path.exists('alexnet_model.pth') and os.path.exists('alexnet_optimizer.pth'):
    print("Загружаем сохранённую модель...")
    alexnet = models.alexnet(weights=None)
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)
    alexnet.load_state_dict(torch.load('alexnet_model.pth', map_location=torch.device('cpu'), weights_only=True))
    optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)
    optimizer.load_state_dict(torch.load('alexnet_optimizer.pth', map_location=torch.device('cpu'), weights_only=True))
    alexnet.eval()
else:
    print("Обучение новой модели...")
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)
    optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
    plot_loss()

random_indices = random.sample(range(len(test_loader.dataset)), 9)
example_data = []
example_targets = []

for idx in random_indices:
    data, target = test_loader.dataset[idx]
    example_data.append(data.unsqueeze(0))
    example_targets.append(target)

example_data = torch.cat(example_data)
example_targets = torch.tensor(example_targets)

alexnet.eval()
with torch.no_grad():
    output = alexnet(example_data)

fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}\nGround Truth: {}".format(output.data.max(1, keepdim=True)[1][i].item(), example_targets[i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()


def test_on_custom_image(image_path):
    img = Image.open(image_path)

    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    img = transform(img).unsqueeze(0)

    alexnet.eval()
    with torch.no_grad():
        output = alexnet(img)
    prediction = output.data.max(1, keepdim=True)[1].item()
    img_show = Image.open(image_path).convert('L').resize((28, 28))
    plt.imshow(img_show, cmap='gray')
    file_name, file_extension = os.path.splitext(image_path)
    plt.title(f"Prediction: {prediction}\nGround Truth: {file_name}")
    plt.axis('off')
    plt.show()


test_on_custom_image(image_path)
