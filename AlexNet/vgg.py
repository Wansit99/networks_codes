import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg_block(num, in_channels, out_channels):
    layers = []
    for i in range(num):
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channel = 1
    for (num_conv, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_conv, in_channel, out_channels))
        in_channel = out_channels

    return nn.Sequential(*conv_blks,
                         nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 4096),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(4096, 10),
                         )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = vgg(conv_arch)

    def forward(self, x):
        x = self.net(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, label)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t lr:'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()), optimizer.state_dict()['param_groups'][0]['lr'])


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 无需计算梯度
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # sum up batch loss
            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, label)
            test_loss += loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # item返回一个python标准数据类型 将tensor转换
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 128
    num_works = 7  # 加载数据集用的cpu核数
    pin_memory = True  # 使用内存更快
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root='./fashionmnist_data/',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Resize(224)])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_works,
        pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root='./fashionmnist_data/',
                              train=False,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Resize(224)])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_works,
        pin_memory=pin_memory
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[12, 24], gamma=0.1)  # 学习率按区间更新

    for epoch in range(1, 10):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == "__main__":
    main()
