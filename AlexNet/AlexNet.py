import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_structure = nn.Sequential(
            # 这里，我们使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过度拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10))

    def forward(self, x):
        x = self.net_structure(x)
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
                       100. * i / len(train_loader), loss.item()),optimizer.state_dict()['param_groups'][0]['lr'])



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 无需计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[12, 24], gamma=0.1)  # 学习率按区间更新

    for epoch in range(1, 10):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == "__main__":
    main()
