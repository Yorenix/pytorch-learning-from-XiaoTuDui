import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter

# 测试Sigmoid激活函数：将输入压缩到(0,1)范围
test_input = torch.tensor([[1, -0.5],
                           [-1, 3]])

# 将2D张量reshape为4D格式: (batch_size, channels, height, width)
test_input = torch.reshape(test_input, (-1, 1, 2, 2))
print("测试输入形状:", test_input.shape)

# 加载CIFAR10数据集，转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 创建数据加载器，批次大小为64
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

# 定义只包含Sigmoid激活函数的神经网络模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.sigmoid1 = nn.Sigmoid()  # Sigmoid激活函数：f(x)=1/(1+e^(-x))
        # self.ReLU1 = ReLU() # 法二

    def forward(self, input):
        # 前向传播：应用Sigmoid激活函数
        output = self.sigmoid1(input)  # 将输入压缩到(0,1)范围
        return output

# 创建模型实例
model = MyModel()

# 创建TensorBoard写入器，用于可视化
writer = SummaryWriter("logs")
step=0

# 遍历数据集进行Sigmoid激活函数测试
for data in dataloader:
    imgs, targets = data  # 获取图像和标签
    writer.add_images("input", imgs, step)  # 记录原始输入图像
    outputs = model(imgs)  # 模型推理（应用Sigmoid激活）
    writer.add_images("output", outputs, step)  # 记录激活后图像
    step+=1  # 步数递增

writer.close()  # 关闭TensorBoard写入器
