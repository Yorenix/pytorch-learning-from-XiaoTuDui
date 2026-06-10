import torchvision
import torch
from torch import nn

# 不使用这个模型了，因为这个模型有一百多G
# train_data = torchvision.datasets.ImageNet(root="./5_data_image_net", train=True,
#                                            download=True,transform=torchvision.transforms.ToTensor())

# pretrained=False：
# 加载模型的架构，但不加载预训练权重,模型参数会被随机初始化,适用于从头开始训练模型
# pretrained=True：
# 加载模型的架构并加载预训练权重,权重是在大型数据集（如ImageNet）上预训练得到的,适用于迁移学习或微调
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

# 根据vgg16这个模型的输出分析，vgg16有1000个分类，我们在模型最后添加一个全连接层，输出10个分类结果
# vgg16_true.add_module("add_Linear", nn.Linear(1000, 10)) 添加到vgg16模型的一级模块中
vgg16_true.classifier.add_module("add_Linear", nn.Linear(1000, 10)) # 添加到vgg16的classifier模块中
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10) # 修改vgg16的classifier模块中第6个全连接层的输出维度为10
print(vgg16_false)
