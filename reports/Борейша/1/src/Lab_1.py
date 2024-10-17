import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
import os

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
image_path = '5.jpg'

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/dataset/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


criterion = nn.CrossEntropyLoss()


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)  # Используем кросс-энтропийную функцию потерь
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    torch.save(network.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if os.path.exists('model.pth'):
    print("Загружаем сохранённую модель...")
    network.load_state_dict(torch.load('model.pth', weights_only=True))
    optimizer.load_state_dict(torch.load('optimizer.pth', weights_only=True))
else:
    print("Обучение новой модели...")
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('CrossEntropyLoss')
    plt.show()


random_indices = random.sample(range(len(test_loader.dataset)), 9)
example_data = []
example_targets = []

for idx in random_indices:
    data, target = test_loader.dataset[idx]
    example_data.append(data.unsqueeze(0))
    example_targets.append(target)

example_data = torch.cat(example_data)
example_targets = torch.tensor(example_targets)

network.eval()
with torch.no_grad():
    output = network(example_data)

fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}\nGround Truth: {}".format(output.data.max(1, keepdim=True)[1][i].item(), example_targets[i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()

# Тест на своей фотографии
image = Image.open(image_path)
inverted_image = ImageOps.invert(image.convert('RGB'))

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),  # Преобразование в черно-белое изображение
    torchvision.transforms.Resize((28, 28)),  # Изменение размера до 28x28
    torchvision.transforms.ToTensor(),  # Преобразование в тензор
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # Нормализация
])
image_tensor = transform(inverted_image)
image_tensor = image_tensor.unsqueeze(0)

network.eval()
with torch.no_grad():
    my_output = network(image_tensor)

plt.imshow(image_tensor[0][0], cmap='gray', interpolation='none')

file_name, file_extension = os.path.splitext(image_path)
plt.title("Prediction: {}\nGround Truth: {}".format(my_output.data.max(1, keepdim=True)[1].item(), file_name))
plt.show()
