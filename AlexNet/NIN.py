import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/root/tf-logs")


def NIN(in_channels, out_channels, kernel_size, stride, padding):
    layer = []
    layer.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding))
    layer.append(nn.ReLU())
    layer.append(
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1))
    layer.append(nn.ReLU())
    layer.append(
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1))
    layer.append(nn.ReLU())

    return nn.Sequential(*layer)


conv_arch = [(1, 96), (96, 255), (256, 384)]

net = nn.Sequential(NIN(1, 96, kernel_size=11, stride=4, padding=0),
                    nn.MaxPool2d(3, stride=2),
                    NIN(96, 255, kernel_size=5, stride=1, padding=2),
                    nn.MaxPool2d(3, stride=2),
                    NIN(255, 384, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(3, stride=2),
                    nn.Dropout(0.5),
                    NIN(384, 10, kernel_size=3, stride=1, padding=1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                    )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(NIN(1, 96, kernel_size=11, stride=4, padding=0),
                                 nn.MaxPool2d(3, stride=2),
                                 NIN(96, 255, kernel_size=5, stride=1, padding=2),
                                 nn.MaxPool2d(3, stride=2),
                                 NIN(255, 384, kernel_size=3, stride=1, padding=1),
                                 nn.MaxPool2d(3, stride=2),
                                 nn.Dropout(0.5),
                                 NIN(384, 10, kernel_size=3, stride=1, padding=1),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten()
                                 )

    def forward(self, x):
        x = self.net(x)
        return x


def train(model, device, train_loader, test_loader, epochs):
    # 如果采用默认初始化权重，很难train的动
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            l = loss(output, label)
            total_loss += l
            l.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t lr:'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                           100. * i / len(train_loader), l.item()), optimizer.state_dict()['param_groups'][0]['lr'])
        test_loss, acc = test(model, device, test_loader)
        total_loss = total_loss / len(train_loader)
        writer.add_scalars('check/Loss', {'Train': total_loss, 'Test': test_loss}, epoch)
        writer.add_scalar('check/Accuracy', acc, epoch)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():  # 无需计算梯度
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # sum up batch loss
            loss = criteria(output, label)
            test_loss += loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # item返回一个python标准数据类型 将tensor转换
            correct += pred.eq(label.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (test_loss / len(test_loader), 100. * correct / len(test_loader.dataset))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 128
    num_works = 12  # 加载数据集用的cpu核数
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

    model = Net()
    epochs = 10
    train(model, device, train_loader, test_loader, epochs)


def model_test():
    X = torch.rand((1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, '\toutput shape:', X.shape)


if __name__ == "__main__":
    # model_test()
    main()
    writer.close()
