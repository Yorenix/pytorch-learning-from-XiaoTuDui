from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# 图像文件路径
img_path = "2_data/train/ants_image/0013035.jpg"
# 使用PIL库打开图像文件，返回PIL.Image对象
img_PIL = Image.open(img_path)

# 创建ToTensor转换器对象
# transforms.ToTensor() 实例化了一个可调用对象
# 这个对象实现了__call__方法，可以像函数一样被调用
tensor_trans = transforms.ToTensor()

# 应用转换：将PIL图像转换为PyTorch张量
# tensor_trans(img_PIL) 实际上调用了 tensor_trans.__call__(img_PIL)
# 转换过程包括：
# 1. 将像素值从[0,255]缩放到[0.0,1.0]
# 2. 将维度从(H,W,C)重新排列为(C,H,W)
# 3. 转换为torch.FloatTensor数据类型
tensor_img = tensor_trans(img_PIL)

writer = SummaryWriter("logs")
writer.add_image("tensorboard_image", tensor_img)
writer.close()
