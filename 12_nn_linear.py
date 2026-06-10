import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.ao.nn.intrinsic import LinearReLU
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR10数据集，转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 创建数据加载器，批次大小为64
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

model = MyModel()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)
