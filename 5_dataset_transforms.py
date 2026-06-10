import torchvision
from torch.utils.tensorboard import writer, SummaryWriter
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 定义Compose工具列表，第一个工具是转换为ToTensor类型
dataset_transforms = transforms.Compose([
    transforms.ToTensor()  # 需要括号来创建实例
])
# 下载好后不会重复下载
train_set = torchvision.datasets.CIFAR10(root='./4_dataset', train=True, transform=dataset_transforms ,download=True)
test_set = torchvision.datasets.CIFAR10(root='./4_dataset', train=False, transform=dataset_transforms , download=True)

# 测试数据集是否加载成功
# print(test_set[0]) # PIL.Image对象
# print(test_set.classes) # 类别标签列表
#
# img, target = test_set[0]
# img.show() # 显示图像
# print(test_set.classes[target]) # 类别标签字符串

write = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    write.add_image("test_set", img, i)
write.close()