"""
Copyright (c) 2017, Gavin Weiguang Ding
All rights reserved

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import torch
import torchvision
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 



# 设定参数
n_epochs = 500             # 设定循环训练集的次数
learning_rate = 0.003     # 学习率。原来是0.01
momentum = 0.5           # 动量，可以被看作是梯度下降过程中的“惯性”
log_interval = 10        # 日志打印间隔
random_seed = 1          # 为使实验可重复，设定随机种子以产生相同序列的随机数
torch.manual_seed(random_seed)


# 定义一个名为 Net 的新类，其基于PyTorch中所有神经网络模块的基类
class Net(nn.Module):
    def __init__(self):                                                   # init 是类的构造函数，self代表类的当前对象
        super(Net, self).__init__()                                       # 调用nn.Module的__init__方法，正确执行初始化
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)                       # 2维卷积层，1个单通道输入，10个特征图输出，卷积核大小是5x5，数字是随机数
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)   
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()                                  # 2维dropout层，随机“关闭”输入单元的一部分（即它们输出为0），利于模型泛化、防止神经网络过拟合
        self.fc1 = nn.Linear(32*5*5, 50)                                  # 全连接层，32*32*32个特征输入，50个特征输出
        self.fc2 = nn.Linear(50, 3)
        self.dropout = nn.Dropout()                                       # 添加这一行定义Dropout
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                         
        # print("conv1 output shape:", x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))                         
        # print("conv2 output shape:", x.shape)
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))        # print("conv3 output shape:", x.shape) # print("卷积层和池化层后的形状: ", x.shape)
        x = x.view(-1, 32*5*5)                                             # 整形x，使其可被全连接层处理；-1表示该维度的大小由数据自动计算
        x = F.relu(self.fc1(x))
        x = self.dropout(x)        
        x = self.fc2(x)
        return x

    
       
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)
