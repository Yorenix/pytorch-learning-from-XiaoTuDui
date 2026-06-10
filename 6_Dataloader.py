# 导入必要的库和模块
import torchvision  # PyTorch视觉库，包含常用的数据集和图像变换
from torch.utils.data import DataLoader  # 数据加载器，用于批量加载数据
from torch.utils.tensorboard import SummaryWriter  # TensorBoard写入器，用于可视化

# 准备测试数据集
# 使用torchvision.datasets.CIFAR10加载CIFAR-10数据集
# root: 数据集存储路径
# train=False: 加载测试集（train=True加载训练集）
# transform: 数据预处理变换，将PIL图像转换为PyTorch张量
test_data = torchvision.datasets.CIFAR10(root='./4_dataset', train=False, transform=torchvision.transforms.ToTensor())

# 创建数据加载器（DataLoader）
# DataLoader负责将数据集分成批次，方便模型训练和评估
test_loader = DataLoader(
    dataset=test_data,        # 要加载的数据集对象
    batch_size=64,            # 每个批次包含的样本数量（类似抽牌数量）
    shuffle=True,             # 是否在每个epoch开始时打乱数据顺序（类似洗牌）
    num_workers=0,            # 用于数据加载的子进程数量（0表示使用主进程）
    drop_last=False           # 是否丢弃最后一个不完整的批次
)

# 测试数据集中第一张图片（注释掉的调试代码）
# img, target = test_data[0]  # 获取第一个样本（图像和标签）
# print(img.shape)           # 打印图像张量的形状
# print(test_data.classes[target])  # 打印对应的类别名称

# 创建TensorBoard写入器，用于记录和可视化数据
# "logs"是TensorBoard日志文件的存储目录
writer = SummaryWriter("logs")

# 初始化步数计数器，用于TensorBoard中的时间轴
step = 0

# 遍历数据加载器中的所有批次
# DataLoader会按批次返回数据，每个批次包含batch_size个样本
for data in test_loader:
    # 解包批次数据：imgs是图像批次，targets是标签批次
    imgs, targets = data
    
    # 将当前批次的图像添加到TensorBoard
    # "test_data": 图像在TensorBoard中的标签名称
    # imgs: 图像批次张量，形状为[batch_size, channels, height, width]
    # step: 当前步数，用于在TensorBoard时间轴上定位
    writer.add_images("test_data", imgs, step)
    
    # 步数加1，准备记录下一个批次
    step += 1

# 关闭TensorBoard写入器，确保所有数据都已写入磁盘
writer.close()