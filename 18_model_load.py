import torch
import torchvision

#加载方式一
model = torch.load("./18_vgg16_method1.pth", weights_only=False)
# print(model)

#加载方式二
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./18_vgg16_method2.pth",weights_only=False))
print(vgg16)
