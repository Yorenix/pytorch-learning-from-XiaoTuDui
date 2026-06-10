import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *

# 加载数据集
train_data = torchvision.datasets.CIFAR10(root='./4_dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./4_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# print(train_data_size)
# print(test_data_size)

# 使用dataloader数据采集器
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

# 创建网络模型
mymodel = Mymodel()
# ***********************************************************************GPU训练
if torch.cuda.is_available():
    mymodel = mymodel.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
# ***********************************************************************GPU训练
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 定义优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epochs = 10

# 添加tensorboard
writer = SummaryWriter(log_dir='./logs')
start_time = time.time()
# 训练循环：遍历每个训练轮次
for i in range(epochs):
    print("-----------第{}轮训练开始了-----------".format(i + 1))

    # 训练步骤开始：遍历训练数据集
    for data in train_dataloader:
        # 获取批次数据：图片和对应的标签
        imgs, targets = data
        # ***********************************************************************GPU训练
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # 前向传播：将图片输入模型，获得预测输出
        outputs = mymodel(imgs)
        # 计算损失：使用损失函数比较预测输出与真实标签
        loss = loss_fn(outputs, targets)
        # 反向传播与优化：
        optimizer.zero_grad()  # 1. 清空上一轮的梯度（避免梯度累积）
        loss.backward()  # 2. 反向传播计算梯度
        optimizer.step()  # 3. 根据梯度更新模型参数
        # 记录训练次数
        total_train_step += 1
        # 打印训练进度（每步都打印，可改为定期打印）
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("所用时间："+str(end_time-start_time))
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 训练结束后用测试集测试，看模型训练是否达到预期
    # 测试开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # ***********************************************************************GPU训练
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = mymodel(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    print("整体测试的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(mymodel, "mymodel_{}.pth".format(i))
    # torch.save(mymodel.state_dict(), "mymodel_{}.pth".format(i)) 官方推荐的保存方式
    print("模型已保存")

writer.close()
