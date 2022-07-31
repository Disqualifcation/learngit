# 导入必要的模块
import numpy as np
import torch
from torchvision.datasets import mnist
# 导入pytorch内置的mnist数据
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 导入预处理模块
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
# 导入nn，及优化器
# 定义一些超参数
train_batch_size = 64  # 定义训练批数据为64
test_batch_size = 128  # 测试批数据为128
learning_rate = 0.01
num_epoches = 20
lr = 0.01  # 学习率为0.01
momentum = 0.5
# 预处理数据
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # 定义预处理函数，将预处理依次放在Compose函数中。
train_dataset = mnist.MNIST('./data', train=True, transform=transform,
                            download=True)  # 下载训练数据，如果./data目录下已有MNIST，可选择Flase
test_dataset = mnist.MNIST('./data', train=False, transform=transform)  # 下载测试数据
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


# 用DataLoader得到生成器，节省内存。通过打乱数据，进行批处理，得到生成器。
# 构建网络，标准模板
class Net(nn.Module):  # 定义Net类，继承于nn.Module类
    def __init__(self, in_dim, n_hidden_1,n_hidden_2,out_dim):  # 初始化对象时默认调用的函数
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim))  # 使用seqyuential构造网络，Sequential（）函数的功能是将网络的层组合在一起，Linear（）函数将输入与隐藏层连接

    def forward(self, x):  # 前馈过程中要用到的函数
        x = F.relu(self.layer1(x))  # 通过relu激活函数激活数据
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x  # 经历三次返回x


# 实例化网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 检验是否有可用的GPU，有则使用，没有则用CPU
model = Net(28 * 28, 300, 100, 10)
# 实例化网络，输入层为28*28层，隐藏层1为300层，2为100层，输出为10
model.to(device)
criterion = nn.CrossEntropyLoss()
# 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# 定义优化器
# 训练模型
losses = []
acces = []
eval_losses = []
eval_acces = []
for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    eval_loss = 0
    eval_acc = 0
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
        losses.append(train_loss / len(test_loader))
    acces.append(train_acc / len(test_loader))
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch, train_loss / len(
        train_loader), train_acc / len(train_loader), eval_loss / len(test_loader), eval_acc / len(test_loader)))
plt.title('trainloss')
plt.plot(np.arange(len(losses)),losses)
plt.legend(['Train Loss'],loc='upper right')
plt.show()