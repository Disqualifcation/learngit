import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch.optim as optim
import numpy as np
# ========================================准备数据========================================
# 定义预处理函数
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#ransfom.Compose可以把一些转换函数组合在一起。②transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))对张量进行归一化：图像有三个通道，每个通道均值0.5，方差0.5。)
# 下载CIFAR10数据，分别下载训练数据和测试数据，并用transforms.Compose函数对数据进行预处理
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
# 得到一个生成器
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
testloader = DataLoader(testset, batch_size=4, shuffle=False)  # 数据分批
"""
①dataloader是一个可迭代对象，可以使用迭代器一样使用。
②用DataLoader得到生成器，这可节省内存。
"""
# 分类类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()


#定义神经网络
class Net(nn.Module):#继承自nn.Module类
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1))  # 卷积层1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=(3, 3), stride=(1, 1))  # 卷积层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层2

        self.fc1 = nn.Linear(1296, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 10)  # 全连接层2

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36 * 6 * 6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x


#实例化神经网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#定义GPU参数
net = Net()  # 实例化网络
net = net.to(device)  # 使用GPU训练

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()#定义损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#定义优化器，model.parameters()：传入模型参数；lr：学习率；momentum：动量，更新缩减和平移参数的频率和幅度，结合当前梯度与上一次更新信息用于当前更新

#训练模型
for epoch in range(10):
    start = time.time()#定义开始的时间
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取训练数据
        inputs, labels = data  # 解包元组
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据放在GPU上
        # 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()  # 清空上一步的残余更新参数值，将神经网络参数梯度降为0
        loss.backward()  # 误差反向传递，计算参数更新值
        optimizer.step()  # 优化梯度，将参数更新值施加到net的parameters上
        # 记录误差
        running_loss += loss.item()
        # 每2000步打印损失
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f, time:%.2f' % (epoch + 1, i + 1, running_loss / 2000, time.time() - start))
            running_loss = 0.0
print('训练完成')

# 预测结果
images, labels = images.to(device), labels.to(device)#将变量GPU加载
outputs = net(images)#输出结果
_, predicted = torch.max(outputs, 1)  # 输出值的最大值作为预测值
print('预测结果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    # 从测试数据中取出数据
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # 模型预测的最大值作为预测值

        total += labels.size(0)  # 统计测试图片个数
        correct += (predicted == labels).sum().item()  # 统计正确预测的图片数
print('10000张测试图片的准确率：%d%%' % (100 * correct / total))

# --------------------每个类别预测准确率--------------------
class_correct = list(0. for i in range(10))  # 正确的类别个数
class_total = list(0. for i in range(10))  # 一共10个类别
with torch.no_grad():
    # 从测试数据中取出数据
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()  # 预测正确的返回True，预测错误的返回False；squeeze将数据转换为一维数据
        for i in range(4):
            label = labels[i]  # 提取标签
            class_correct[label] += c[i].item()  # 预测正确个数
            class_total[label] += 1  # 总数
for i in range(10):
    print('%5s的准确率:%2d%%' % (classes[i], 100 * class_correct[i] / class_total[i]))

