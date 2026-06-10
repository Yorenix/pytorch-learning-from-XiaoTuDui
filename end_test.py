import torch
import torchvision
from PIL import Image
from torch import nn


# ================= 定义模型结构 =================
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)


# ================= 读取图片 =================
image_path = "./3_images/airplane.png"

image = Image.open(image_path)

image = image.convert("RGB")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)

# 增加batch维度
image = torch.reshape(image, (1, 3, 32, 32))


# ================= GPU =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 创建模型 =================
model = Mymodel()

# 加载参数
model.load_state_dict(torch.load("./mymodel_28.pth"))

# 放到GPU
model = model.to(device)

image = image.to(device)


# ================= 测试 =================
model.eval()

with torch.no_grad():

    output = model(image)

print(output)

print(output.argmax(1))