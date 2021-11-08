import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
from torchvision import datasets, transforms
from ResNet_pre import *

trainss = pd.read_csv('leaf/train.csv')
test = pd.read_csv('leaf/test.csv')

label = pd.get_dummies(trainss, columns=['label'])
labels = label[:].values
labels = np.array(labels[:, 1:])

pred = np.argmax(labels, axis=1)
label_list = label.columns.values[1:]
label_list = [i[6:] for i in label_list]

rate = 0.9

root = 'leaf/classify-leaves/images/'
root_dir = os.listdir(root)
root_dir.sort(key=lambda x: int(x[:-4]))
train_img_dir = root_dir[:int(len(pred) * rate)]
test_img_dir = root_dir[int(len(pred) * rate):len(pred)]
train_label = pred[:int(len(pred) * rate)]
test_label = pred[int(len(pred) * rate):len(pred)]


class LeafData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, labels, transform=None, train=True):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件
        self.images.sort(key=lambda x: int(x[:-4]))
        self.images = self.images[:len(labels)]
        self.labels = labels
        self.train = train
        self.len = len(labels)
        self.gap = self.len - int(len(labels) * rate)
        self.add = - (len(labels) * rate)
        self.target_transform = transform

        if train:
            self.images = self.images[:int(len(labels) * rate)]
            self.labels = self.labels[:int(len(labels) * rate)]
        else:
            self.images = self.images[int(len(labels) * rate):]
            self.labels = self.labels[int(len(labels) * rate):]

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        if self.train:
            index %= len(self.images)
        else:
            index %= len(self.images)

        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = cv2.imread(img_path)  # 读取该图片
        # 加上这句话 才是[128, 3, 224, 224]
        # img = torch.from_numpy(img).permute(2, 0, 1)
        label_idx = int(img_path.split('/')[-1].split('.')[0])

        label = self.labels[
            int(index)]

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == "__main__":

    batch_size = 16
    num_works = 6  # 加载数据集用的cpu核数
    pin_memory = True  # 使用内存更快
    trains = LeafData(
        root_dir=root,
        labels=pred,
        transform=transforms.ToTensor(),
        train=True)  # 初始化类，设置数据集所在路径以及变换
    train_loader = torch.utils.data.DataLoader(
        trains,
        batch_size=batch_size,
        num_workers=num_works,
        shuffle=True,
        pin_memory=pin_memory)  # 使用DataLoader加载数据

    test = LeafData(
        root_dir=root,
        labels=pred,
        transform=transforms.ToTensor(),
        train=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True,
                             num_workers=num_works,
                             pin_memory=pin_memory)  # 使用DataLoader加载数据

    model = Net()
    # pretrained_dict = torch.load('resnet18.pth')
    # model_dict = model.state_dict()
    # pretrained_dict = {'BackBone.' + k: v for k, v in pretrained_dict.items() if
    #                    ('BackBone.' + k in model_dict and 'fc' not in k)}
    #
    # # for k, v in pretrained_dict.items():
    # #     print(k)
    #
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    epochs = 15
    lr = 1
    num_gpus = 1
    save_dir = 'leaf_params'
    if num_gpus > 1:
        # mul gpu
        devices = [try_gpu(i) for i in range(num_gpus)]
        print(devices)
        train_mul(model, devices, train_loader, test_loader, epochs, lr, save_dir)
    else:
        # single gpu
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(model, device, train_loader, test_loader, epochs, lr, save_dir)

    writer.close()

#     model_test()

