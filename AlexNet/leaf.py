import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader,Dataset
import cv2
import torch
from torchvision import datasets, transforms
from ResNet import *

trainss = pd.read_csv('./leaf/train.csv')
test = pd.read_csv('./leaf/test.csv')

label = pd.get_dummies(trainss, columns=['label'])
labels = label[:].values
labels = np.array(labels[:,1:])

pred = np.argmax(labels, axis=1)
label_list = label.columns.values[1:]
label_list = [i[6:] for i in label_list]

root = './leaf/classify-leaves/images'
root_dir = os.listdir(root)
root_dir.sort(key=lambda x: int(x[:-4]))
train_img_dir = root_dir[:int(len(pred)*0.9)]
test_img_dir = root_dir[int(len(pred)*0.9):len(pred)]
train_label = pred[:int(len(pred)*0.9)]
test_label = pred[int(len(pred)*0.9):len(pred)]

pred_dir = root_dir[len(pred):]

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
        self.gap = self.len - (len(labels) * 0.9)
        self.add = self.gap - self.len
        self.target_transform = transform

        if train:
            self.images = self.images[:int(len(labels) * 0.9)]
            self.labels = self.labels[:int(len(labels) * 0.9)]
        else:
            self.images = self.images[int(len(labels) * 0.9):]
            self.labels = self.labels[int(len(labels) * 0.9):]

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        if self.train:
            index = index
        else:
            if index > self.gap:
                index += self.add

        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = cv2.imread(img_path)  # 读取该图片
        # 加上这句话 才是[128, 3, 224, 224]
        #img = torch.from_numpy(img).permute(2, 0, 1)
        label_idx = int(img_path.split('\\')[-1].split('.')[0])

        if self.train:
            pass
        else:
            if int(label_idx) > self.gap:
                label_idx += self.add

        label = self.labels[
            int(label_idx)]

        if self.transform is not None:
            img = self.transform(np.array(img))  # 是否进行transform
        return img, label


class PreData(Dataset):  # 继承Dataset
    def __init__(self, root, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.images = root_dir
        self.transform = transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        tmp = torch.tensor([1])
        image_dir = self.images[index]  # 根据索引index获取该图片
        image_dir = os.path.join(root, image_dir)
        img = cv2.imread(image_dir)  # 读取该图片

        if self.transform is not None:
            img = self.transform(np.array(img))  # 是否进行transform
        return img, tmp



def predict(model, device, root, root_dir, batch_size):

    pre = PreData(root, root_dir, transform=transforms.ToTensor())
    pre_loader = torch.utils.data.DataLoader(pre, batch_size=batch_size,
                                               num_workers=7,
                                               shuffle=False,
                                               pin_memory=True)  # 使用DataLoader加载数据

    model.to(device)
    model.eval()
    pred = []
    with torch.no_grad():
        for i, (data, label) in enumerate(pre_loader):
            data = data.to(device)
            output = model(data)
            output = output.argmax(dim=1, keepdim=False)
            output = output.tolist()
            pred = pred + output

    return pred

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

if __name__ == "__main__":


    batch_size = 32
    num_works = 7  # 加载数据集用的cpu核数
    pin_memory = True  # 使用内存更快
    trains = LeafData(root_dir=root, labels=pred, transform=transforms.ToTensor(), train=True)  # 初始化类，设置数据集所在路径以及变换
    train_loader = torch.utils.data.DataLoader(trains, batch_size=batch_size,
                                               num_workers=num_works,
                                               shuffle=True,
                                               pin_memory=pin_memory)  # 使用DataLoader加载数据

    test = LeafData(root_dir=root, labels=pred, transform=transforms.ToTensor(), train=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True,
                             num_workers=num_works,
                             pin_memory=pin_memory)  # 使用DataLoader加载数据

    model = Net()
    epochs = 10

    num_gpus = 1

    if num_gpus >1:
        # mul gpu
        devices = [try_gpu(i) for i in range(num_gpus)]
        train_mul()
    else:
        # single gpu
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(model, device, train_loader, test_loader, epochs)




    # model_test()
    # model.load_state_dict(torch.load('./leaf/para'))
    # result = predict(model, device, root, pred_dir, batch_size)
    # result = [label_list[i] for i in result]
    #
    # pred_label = pd.DataFrame({'label':result})
    # result = pd.concat([test, pred_label], axis=1)
    # print(result.head())
    # result.to_csv('result.csv',index=None)
    # tmp = torch.rand([32, 3, 224, 224])
    # result = model(tmp)
    # result = result.argmax(dim=1, keepdim=False)
    # print(result.shape)




