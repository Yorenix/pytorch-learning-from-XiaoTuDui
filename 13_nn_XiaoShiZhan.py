import torch
import torchvision
from networkx.generators.classic import kneser_graph
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR10数据集，转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        #               方法一：分别定义各个卷积、池化、展开、线性。。。
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=1024, out_features=64)
        # self.linear2 = nn.Linear(in_features=64, out_features=10)
        #              方法二：使用Sequential
        self.model1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Flatten(),
                                    nn.Linear(in_features=1024, out_features=64),
                                    nn.Linear(in_features=64, out_features=10)
                                    )

    def forward(self, x):
        #      方法一
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        #      方法二
        return self.model1(x)
        return x

Model = MyModel()
print(Model)
# 测试
input = torch.ones(64,3,32,32)
output = Model(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(Model,input)
writer.close()
