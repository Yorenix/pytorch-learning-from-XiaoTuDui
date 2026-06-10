from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "2_data/train/ants_image/6240329_72c01e663e.jpg"
img_PIL = Image.open(img_path)
img_arr = np.array(img_PIL)

writer.add_image("test", img_arr, 2, dataformats='HWC')


for i in range(100):
    writer.add_scalar("y=x^3", i**3, i)


writer.close()

print("done")