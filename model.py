import torch
from torch import nn

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第二层
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第三层
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # 全连接层
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            # 防止过拟合
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    mymodel = Mymodel()
    input = torch.ones(64,3,32,32)
    output = mymodel(input)
    print(output.shape)
