# 导入必要的库
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 注释掉的示例：手动定义5x5输入矩阵进行最大池化测试
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (-1, 1, 5, 5))

# 加载CIFAR10数据集，转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 创建数据加载器，批次大小为64
dataloader = DataLoader(dataset,batch_size=64)

# 定义包含最大池化层的神经网络模型
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        # 定义最大池化层：3x3池化窗口，使用向上取整模式
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        # 前向传播：应用最大池化操作
        output = self.maxpool1(input)
        return output

# 创建模型实例
Module = MyModule()

# 创建TensorBoard写入器，用于可视化
writer = SummaryWriter("logs")
step=0

# 遍历数据集进行最大池化操作
for data in dataloader:
    imgs, targets = data  # 获取图像和标签
    writer.add_images("input",imgs,step)  # 记录输入图像
    output = Module(imgs)  # 应用最大池化
    writer.add_images("output",output,step)  # 记录池化后图像
    step+=1  # 步数递增

writer.close()  # 关闭TensorBoard写入器