import torch
import torchvision.datasets
import torch.nn as nn

# 加载CIFAR10数据集，转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(root='./4_dataset', train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=3)

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

# 创建模型和损失函数
model = MyModel()
loss = nn.CrossEntropyLoss()

# 创建优化器（随机梯度下降），设置学习率为0.01
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练20个epoch（完整遍历数据集20次）
for epoch in range(20):
    # 初始化当前epoch的累计损失
    runing_loss = 0.0
    
    # 遍历数据加载器中的所有批次
    for data in dataloader:
        # 解包批次数据：图像和对应的标签
        imgs, targets = data
        
        # 前向传播：模型对输入图像进行预测
        outputs = model(imgs)
        
        # 计算损失：比较模型输出与真实标签
        result_loss = loss(outputs, targets)
        
        # 梯度清零：清除上一批次的梯度，防止梯度累积
        optim.zero_grad()
        
        # 反向传播：计算损失相对于模型参数的梯度
        result_loss.backward()
        
        # 参数更新：根据梯度使用优化器更新模型权重
        optim.step()
        
        # 累计损失：将当前批次的损失加到总损失中
        runing_loss+=result_loss
    
    # 打印当前epoch的总损失
    print(runing_loss)