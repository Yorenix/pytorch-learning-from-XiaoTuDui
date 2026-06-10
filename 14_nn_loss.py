import torch
from torch.nn import L1Loss, MSELoss

# 创建输入和目标张量
input = torch.tensor([1,2,3], dtype=torch.float)
target = torch.tensor([1,2,5], dtype=torch.float)

# 调整形状为(batch, channel, height, width)
input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target,(1,1,1,3))

# L1损失(绝对值损失)，reduction='sum'表示求和
loss = L1Loss(reduction='sum')
output = loss(input, target)
print(output)

# MSE损失(均方误差)，默认reduction='mean'
loss = MSELoss()
output = loss(input, target)
print(output)