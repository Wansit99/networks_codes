import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms





def MyDataloader(train_aug, test_aug, batch_size, num_works, pin_memory=True, shuffle=True):
    TRAIN_DATA_PATH = "../data/fer2013/train"
    TEST_DATA_PATH = "../data/fer2013/test/"
    train_aug = transforms.Compose([
        transforms.Resize(256),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()]
    )
    test_aug = transforms.Compose([
        transforms.Resize(256),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()]
    )

    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_aug)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_works, pin_memory=pin_memory)  # 使用DataLoader加载数据

    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_aug)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_works, pin_memory=pin_memory)  # 使用DataLoader加载数据

    return train_loader, test_loader

