import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 创建数据加载器，批次大小为64
dataloader = DataLoader(dataset,batch_size=64)

# 定义自定义神经网络模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        # 定义卷积层：输入通道3，输出通道6，卷积核3x3
        self.conv1 = nn.Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
        # 前向传播：应用卷积操作
        x = self.conv1(x)
        return x

# 创建模型实例
Model = MyModel()

# 创建TensorBoard写入器，用于可视化
writer = SummaryWriter("logs")
step = 0

# 遍历数据集进行训练/测试
for data in dataloader:
    imgs, targets = data  # 获取图像和标签
    output = Model(imgs)  # 模型推理
    print(imgs.shape)  # 打印输入形状
    print(output.shape)  # 打印输出形状
    
    # 将输入图像写入TensorBoard
    writer.add_images("input",imgs,step)
    
    # 重塑输出以适应TensorBoard显示（6通道->3通道）
    output = torch.reshape(output,(-1,3,30,30))
    
    # 将输出图像写入TensorBoard
    writer.add_images("output",output,step)
    step += 1  # 步数递增