import torchvision as tvn
import torchvision.models as models
import torchvision.transforms as tfs
import torch.utils.data as data
from torchvision.datasets import CIFAR100
import torch.optim as opt
import torch.nn as nn
import torch
from tqdm import tqdm
from PIL import Image

lr = 0.001
epochs = 3
batch_size = 32


def train(model, data_train, loss_f, optz):
    for i in range(epochs):
        loss_mean = 0
        lm_count = 0

        train_tqdm = tqdm(data_train, leave=True)
        for x_data, y_data in train_tqdm:
            predict = model(x_data)
            loss = loss_f(predict, y_data)

            optz.zero_grad()
            loss.backward()
            optz.step()

            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

            train_tqdm.set_description(f"Epoch[{i + 1}/{epochs}], loss_mean={loss_mean:.3f}")


def test(model, data_test):
    correct = 0
    total = 0

    test_tqdm = tqdm(data_test, leave=True)
    for x_test, y_test in test_tqdm:
        with torch.no_grad():
            p = model(x_test)
            _, p = torch.max(p.data, 1)
            correct += (p == y_test).sum().item()
            total += y_test.size(0)

    accuracy = correct / total
    return accuracy * 100


def main():

    transform_train = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.Resize(224),
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = tfs.Compose([
        tfs.Resize(224),
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    check_image(transform_test)
    train_dt = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dt = CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    data_train = data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    data_test = data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)

    model_squeezeNet = models.squeezenet1_1(pretrained=True)
    model_squeezeNet.classifier[1] = nn.Conv2d(512, 100, kernel_size=1)
    model_squeezeNet.num_classes = 100
    torch.nn.init.kaiming_uniform_(model_squeezeNet.classifier[1].weight)

    for param in model_squeezeNet.features.parameters():
        param.requires_grad = False

    optimizer = opt.RMSprop(model_squeezeNet.parameters(), lr=0.0001, weight_decay=1e-4)
    loss_f = nn.CrossEntropyLoss()

    model_squeezeNet.train()
    train(model_squeezeNet, data_train, loss_f, optimizer)

    torch.save(model_squeezeNet, 'nn.pth')

    model_squeezeNet.eval()
    print("Test Accuracy:", test(model_squeezeNet, data_test))



def check_image(trf):
    image = Image.open('img.png').convert('RGB')
    n_image = trf(image).unsqueeze(0)

    model = torch.load('nn.pth')
    model.eval()

    with torch.no_grad():
        output = model(n_image)
        probabilities = nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    class_names = CIFAR100('./data').classes
    print(print(f"Predicted class: {predicted_class}, Class name: {class_names[predicted_class]},"
                f" Confidence: {confidence:.2f}"))


if __name__ == "__main__":
    main()
