import torch
import torchvision.datasets
import torch.nn as nn

# 加载CIFAR10数据集，转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=25)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Flatten(),
                                    nn.Linear(in_features=1024, out_features=64),
                                    nn.Linear(in_features=64, out_features=10)
                                    )
    def forward(self, x):
        return self.model1(x)

model = MyModel()
loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    outputs = model(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)
    # print(outputs)
    # print(targets)


