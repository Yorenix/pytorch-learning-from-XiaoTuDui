from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "3_images/1.jpg"
img_PIL = Image.open(img_path).convert('RGB')  # 确保转换为RGB格式，避免RGBA通道问题.convert('RGB')  # 确保转换为RGB格式，避免RGBA通道问题
writer = SummaryWriter("logs")

#to_tensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_PIL)
writer.add_image("ToTensor", img_tensor)

#normalize
#归一化计算公式``output[channel] = (input[channel] - mean[channel]) / std[channel]``,即每个通道的像素值减去该通道的均值，再除以该通道的标准差
print(img_tensor[0][318][985])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][318][985])
writer.add_image("Normalize", img_norm)

#resize
print(img_PIL.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img_PIL)
print(img_resize.size)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize,0)

# compose 用于将多个transform组合起来，实现对图片的多个操作
trans_resize_2 = transforms.Resize((512,512))
# 列表中的transform按顺序执行，第一个是将图片resize为512*512，第二个是将图片转换为tensor，注意参数顺序不能反，因为resize的参数是PIL而不是tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img_PIL)
writer.add_image("Resize", img_resize_2,1)

# RandomCrop
trans_random_crop = transforms.RandomCrop((256,512))
trans_compose_2 = transforms.Compose([trans_random_crop,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img_PIL)
    writer.add_image("RandomCrop", img_crop,i)

writer.close()

