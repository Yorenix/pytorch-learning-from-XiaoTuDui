import torch
# 导入神经网络模块，包含各种神经网络层和工具
from torch import nn

# 定义自定义神经网络模块类，继承自nn.Module基类
# 所有PyTorch神经网络模块都必须继承nn.Module
class MyModule(nn.Module):
    # 初始化方法，在创建模块实例时调用
    def __init__(self):
        # 调用父类nn.Module的初始化方法
        # 这是必须的步骤，用于正确初始化模块的基础结构
        super().__init__()

    # 前向传播方法，定义数据如何通过模块
    # 这是神经网络的核心计算逻辑
    def forward(self, input):
        # 简单的计算示例：将输入值加1
        # 在实际神经网络中，这里会包含复杂的层间计算
        output = input + 1
        # 返回计算结果
        return output

# 创建自定义模块的实例
# 实例化后，M就是一个完整的神经网络模块
M = MyModule()

# 创建测试输入张量
# torch.tensor(1.0) 创建一个值为1.0的标量张量
x = torch.tensor(1.0)

# 使用模块处理输入数据
# M(x) 实际上调用了 M.forward(x)
# 这是PyTorch的语法糖，让模块可以像函数一样被调用
print(M(x))