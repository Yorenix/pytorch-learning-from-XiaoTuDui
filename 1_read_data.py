"""
读取数据集
"""
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    """
    自定义PyTorch数据集类，用于加载图像数据
    Dataset 是一个抽象基类，需要重写 __getitem__ 和 __len__ 方法
    用于将数据组织成PyTorch可以处理的格式
    """
    def __init__(self, root_dir, label_dir):
        """
        构造函数，初始化数据集
        Args:
            root_dir (str): 数据集根目录路径
            label_dir (str): 标签目录名称，同时也是类别名称
        """
        # 存储根目录和标签目录
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 构建完整的图像目录路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取该目录下所有图像文件的文件名列表
        self.img_path = os.listdir(self.path)
    def __getitem__(self, idx):
        """
        根据索引获取单个数据样本
        Args:
            idx (int): 数据样本的索引
        Returns:
            tuple: (图像对象, 标签字符串)
        """
        # 根据索引获取图像文件名
        img_name = self.img_path[idx]
        # 构建完整的图像文件路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 使用PIL打开图像文件
        img = Image.open(img_item_path)
        # 标签就是目录名称（假设每个目录对应一个类别）
        label = self.label_dir
        # 返回图像和标签（注意：这里应该返回img和label）
        return img, label
    def __len__(self):
        """
        返回数据集中样本的总数量
        Returns:
            int: 数据集中图像文件的数量
        """
        return len(self.img_path)

root_dir = "1_dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
"""
train_dataset是一个数据集，train_dataset[0]返回的是第0个样本，返回的是一个元组(img对象,标签字符串）
，执行img, label = train_dataset[0]后，要显示图片使用img.show()方法，要显示标签使用print(label)方法
"""
