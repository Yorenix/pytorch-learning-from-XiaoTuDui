import torchvision
import torch
from sympy import false

vgg16 = torchvision.models.vgg16(weights=None)

#保存方式一:保存模型结构+模型参数
torch.save(vgg16, "./18_vgg16_method1.pth")

#保存方式二：仅保存模型参数(官方推荐)
torch.save(vgg16.state_dict(), "./18_vgg16_method2.pth")

