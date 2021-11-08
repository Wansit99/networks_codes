import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
from torchvision import datasets, transforms
from ResNet import *
import torchvision

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





def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == "__main__":

    batch_size = 128
    num_works = 4  # 加载数据集用的cpu核数
    pin_memory = True  # 使用内存更快

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
        torchvision.transforms.ToTensor()])

    train_loader = torch.datasets.CIFAR10(root='cifir', train=True, download=True, transform=train_augs)
    test_loader = torch.datasets.CIFAR10(root='cifir', train=False, download=False, transform=transforms.ToTensor())

    model = Net()
    epochs = 15
    lr = 0.1
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

