import torch
import torch.nn.functional as F

# 定义5x5输入矩阵
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 定义3x3卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 将2D张量reshape为4D格式: (batch_size, channels, height, width)
input = torch.reshape(input, (1, 1, 5, 5))  # 批次1, 通道1, 高5, 宽5
kernel = torch.reshape(kernel, (1, 1, 3, 3))  # 输出通道1, 输入通道1, 高3, 宽3

print("输入形状:", input.shape)
print("卷积核形状:", kernel.shape)

# 执行卷积操作: 输入, 卷积核, 步长=1
output = F.conv2d(input, kernel, stride=1)
print("卷积结果:")
print(output)
output = F.conv2d(input, kernel,stride=2)
print(output)
output = F.conv2d(input, kernel,stride=1,padding=1)
print(output)