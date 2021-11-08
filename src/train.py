# coding = utf-8
# @Time   : 20-11-17
# @Author : 郭冰洋
# @File   : train.py
# @Cont   : 分类训练

import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from utils.utils import train,validate,creatdir
from models.ResNet import ResNet50
import torchvision
from datasets.emotion import MyDataloader
from utils.utils import try_gpu
from torch.utils.tensorboard import SummaryWriter

# 参数设置
parser = argparse.ArgumentParser()
# 数据集路径
parser.add_argument('--filename', type=str, default= '../logs', help='whether to train.txt')
# 模型及数据存储路径
parser.add_argument('--checkpoint_dir', type=str, default='../model_para', help='directory where model checkpoints are saved')
# 批次
parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
# cuda设置
parser.add_argument('--cuda', type=str, default="0", help='whether to use cuda if available')
# CPU载入数据线程设置
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
# 暂停设置
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
# 迭代次数
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
# 起始次数（针对resume设置）
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
# 显示结果的间隔
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
# 确认参数，并可以通过opt.xx的形式在程序中使用该参数
parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus used')
# gpu使用数量
parser.add_argument('--logs', type=str, default='../tf-logs', help='number of gpus used')
# gpu使用数量
opt = parser.parse_args()

# tensorboard日志存放地址
writer = SummaryWriter(opt.logs)


if __name__ == '__main__':
    # 创建存储及日志文件
    creatdir(opt.checkpoint_dir)
    # 数据增强
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
        torchvision.transforms.ToTensor()])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])


    # 载入数据集
    train_loader, test_loader = MyDataloader(train_augs, test_augs, opt.batch_size, opt.n_cpu)

    # 网络模型的选择
    model = ResNet50(num_classes=176)

    # 如果采用默认初始化权重，很难train的动
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    model.apply(init_weights)

    # 确定devices
    if opt.num_gpus > 1:
        # mul gpu
        devices = [try_gpu(i) for i in range(opt.num_gpus)]
        model = nn.DataParallel(model, device_ids=devices)

    else:
        # single gpu
        use_cuda = torch.cuda.is_available()
        devices = torch.device("cuda" if use_cuda else "cpu")
        model.to(devices)



    #优化器选择
    # Adam优化
    # optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    # SGD优化
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)

    # 损失函数
    criterion = CrossEntropyLoss()

    # 学习率衰减设置
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 最佳准确率置0
    best_precision = 0

    # 设置损失
    lowest_loss = 10000

    # 训练
    for epoch in range(opt.epochs):
        # 训练
        train_acc, train_loss, test_acc, test_loss = train(model, devices, train_loader, test_loader,
          criterion, optimizer, epoch, opt.print_interval, opt.filename, gpus=False)

        # 在tensorboard中写入loss及acc
        writer.add_scalars(
            'check/Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_scalars('check/Accuracy',
                          {'Train': train_acc, 'Test': test_acc}, epoch)


        # 在日志文件中记录每个epoch的训练精度和损失
        with open(opt.checkpoint_dir+'_' + 'record.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, train_Precision: %.8f, train_Loss: %.8f\n' % (epoch, train_acc, train_loss))
        # 在日志文件中记录每个epoch的验证精度和损失
        with open(opt.checkpoint_dir +'_'+ 'record_val.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, test_acc, test_loss))
            # 记录最高精度与最低loss
            is_best = test_acc > best_precision
            is_lowest_loss = test_loss < lowest_loss
            best_precision = max(test_acc, best_precision)
            lowest_loss = min(test_loss, lowest_loss)
            print('--'*30)
            print(' * Accuray {acc:.3f}\t'.format(acc=test_acc), '(Previous Best Acc: %.3f)' % best_precision,
                  ' * Loss {loss:.3f}\t'.format(loss=test_loss), 'Previous Lowest Loss: %.3f)' % lowest_loss)
            print('--' * 30)
            # 保存最新模型
            save_path = os.path.join(opt.checkpoint_dir,'checkpoint.pth')
            torch.save(model.state_dict(),save_path)
            # 保存准确率最高的模型
            best_path = os.path.join(opt.checkpoint_dir,'best_model.pth')
            if is_best:
                shutil.copyfile(save_path, best_path)
            # 保存损失最低的模型
            lowest_path = os.path.join(opt.checkpoint_dir, 'lowest_loss.pth')
            if is_lowest_loss:
                shutil.copyfile(save_path, lowest_path)