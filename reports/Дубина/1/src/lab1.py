import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os

# CIFAR-100 class labels
cifar100_classes = torchvision.datasets.CIFAR100('dataset/', train=True, download=True).classes

n_epochs = 50
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load CIFAR-100 data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('dataset/', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          (0.5071, 0.4867, 0.4408),  # CIFAR-100 mean
                                          (0.2675, 0.2565, 0.2761)   # CIFAR-100 std
                                      )
                                  ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('dataset/', train=False, download=True,
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          (0.5071, 0.4867, 0.4408),
                                          (0.2675, 0.2565, 0.2761)
                                      )
                                  ])),
    batch_size=batch_size_test, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)  # Changed to 3 input channels for RGB
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusted input size for FC layer
        self.fc2 = nn.Linear(128, 100)  # Output classes changed to 100 for CIFAR-100

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

criterion = nn.CrossEntropyLoss()


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
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
    network.load_state_dict(torch.load('model.pth'))
    optimizer.load_state_dict(torch.load('optimizer.pth'))
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


# Display example predictions with class names
random_indices = random.sample(range(len(test_loader.dataset)), 3)
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
    predictions = output.data.topk(5, 1)[1]  # Get top 5 predictions
    example_targets = example_targets.tolist()

fig = plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1)
    plt.tight_layout()
    plt.imshow(example_data[i].permute(1, 2, 0), interpolation='none')
    pred_class = cifar100_classes[predictions[i][0].item()]
    true_class = cifar100_classes[example_targets[i]]
    plt.title(f"Prediction: {pred_class}\nGround Truth: {true_class}")
    plt.xticks([])  # Hide axes
    plt.yticks([])

    # Plot top-5 predictions bar chart
    plt.subplot(3, 3, i * 3 + 2)
    top5_classes = [cifar100_classes[predictions[i][j].item()] for j in range(5)]
    top5_probs = output[i][predictions[i]].exp().tolist()  # Convert log probs to normal
    plt.barh(top5_classes, top5_probs)
    plt.xlim(0, 1)
    plt.xlabel('Probability')
    plt.title(f"Top-5 Predictions for {true_class}")

plt.show()
