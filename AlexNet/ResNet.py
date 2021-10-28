import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary

writer = SummaryWriter("/root/tf-logs")


class Residual(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            use1x1conv=False,
            stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        if use1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                padding=0)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blocks = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blocks.append(Residual(input_channels, output_channels, use1x1conv=True, stride=2))
        else:
            blocks.append(Residual(output_channels, output_channels))
    return blocks

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512,176))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = net

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
        writer.add_scalars(
            'check/Loss', {'Train': total_loss, 'Test': test_loss}, epoch)
        writer.add_scalar('check/Accuracy', acc, epoch)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():  # 无需计算梯度
        for i, (data, label) in enumerate(test_loader):
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
    return (test_loss / len(test_loader), 100. *
            correct / len(test_loader.dataset))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 64
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

    model = Net()
    epochs = 10
    train(model, device, train_loader, test_loader, epochs)


def model_test():
    X = torch.rand((1, 3, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, '\toutput shape:', X.shape)

def print_model():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    summary(model, ((1, 224, 224)))

if __name__ == "__main__":
    #model_test()
    main()
    #print_model()
    #print(Net())
    writer.close()
