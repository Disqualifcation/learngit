#导入必要的包
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
# 在当前目录，创建不存在的目录ave_samples
sample_dir = 'ave_samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
#定义一些超参数
image_size = 784#定义图片的尺寸
h_dim=400
z_dim=20
num_epochs=30 #循环次数为30
batch_size=128 #批数据为128
learning_rate=0.001 #学习率为0.001
#对数据集进行预处理，把数据集转换为循环，可批量加载的数据集
dataset=torchvision.datasets.MNIST(root='data',train=True,transform=transforms.ToTensor(),download=False)#下载MNIST训练集，当train=True时，即下载训练集；当train=False时，则下载测试数据集。transforms.ToTensor()：在做数据归一化之前必须要把PIL Image转成Tensor。download确认数据集是否需要下载，当download=True,则需要下载；反之，不需要下载。
#数据加载
data_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)#迭代器，将数据dataset输入，打乱（shuffle=True）数据，并按批数据进行批处理。
#构建AVE模型，主要由Encode和Decode两部分组成。
class VAE(nn.Module):#定义AVE模型,继承于nn.Module类
    def __init__(self,image_size=784,h_dim=400,z_dim=20):
        super(VAE,self).__init__()#固定继承
        self.fc1=nn.Linear(image_size,h_dim)#设置网络的全连接层，image_size为输入的二维张量的大小，h_dim为此刻输出的二维张量的大小。
        self.fc2=nn.Linear(h_dim,z_dim)
        self.fc3=nn.Linear(h_dim,z_dim)
        self.fc4=nn.Linear(z_dim,h_dim)
        self.fc5=nn.Linear(h_dim,image_size)
    def encode(self,x):
        h=F.relu(self.fc1(x))
        return self.fc2(h),self.fc3(h)
    def reparameterize(self,mu,log_var):
        std=torch.exp(log_var/2)
        eps=torch.randn_like(std)
        return mu+eps*std
    def decode(self,z):
        h=F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    def forward(self,x):
        mu,log_var=self.encode(x)
        z = self.reparameterize(mu,log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
model=VAE().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    for i, (x, _) in enumerate(data_loader):
        # 前向传播
        model.zero_grad()
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)

        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 反向传播及优化器
        loss = reconst_loss + kl_div    # 两者相加得总损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))
    with torch.no_grad():
        z=torch.randn(batch_size,z_dim).to(device)
        out=model.decode(z).view(-1,1,28,28)
        save_image(out,os.path.join(sample_dir,'sampled-{}.png'.format(epoch+1)))
        out,_,_=model(x)
        x_comcat=torch.cat([x.view(-1,1,28,28),out.view(-1,1,28,28)],dim=3)
        save_image(x_comcat,os.path.join(sample_dir,'reconst-{}.png'.format(epoch+1)))
reconsPath='./ave_samples/reconst-30.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)
plt.axis('off')
plt.show()
genPath = './ave_samples/sampled-30.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)
plt.axis('off')
plt.show()
