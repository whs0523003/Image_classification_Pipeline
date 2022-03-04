import argparse
import numpy as np
import cv2
import os

# 计算自定义数据集的mean和std要把所有数据（训练，验证，测试）都放一起计算。因为本来的思想是先预处理完数据后再分训练，验证，测试集
# 所以先把所有数据都放到一块（加载到一个列表里），再进行计算

img_h, img_w = 32, 32  # 数据集图片被resize后的宽高       <=========需手动更改成和训练图片一样大小

# 自定义数据集路径
train_imgs_path = r'D:\datasets\imagewoof-320\train'  # <=========需手动更改
test_imgs_path = r'D:\datasets\imagewoof-320\val'  # <=========需手动更改

means, stds = [], []  # 用于存放mean和std
img_list = []  # 用于存放所有图片的路径
img_list_resized = []  # 用于存放所有处理后的图片

# 获得训练集
for root, dirs, imgs in os.walk(train_imgs_path):
    # 遍历类别
    for sub_dir in dirs:
        img_names = os.listdir(os.path.join(root, sub_dir))
        # 遍历图片
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(root, sub_dir, img_name)
            # 数据格式为[(path_img1), (path_img2) ...]
            img_list.append(path_img)

# # 获得测试集
# imgs_test = os.listdir(test_imgs_path)
# for img in imgs_test:
#     path_img = os.path.join(test_imgs_path, img)
#     img_list.append(path_img)

# 获取数据集总长度
length = len(img_list)


i = 0
for item in img_list:
    img = cv2.imread(item)
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]  # 在最后新添加一个维度，用于后边的batch size
    img_list_resized.append(img)
    i += 1
    print(i, '/', length)

imgs = np.concatenate(img_list_resized, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stds.append(np.std(pixels))

# BGR --> RGB, CV读取的需要转换，PIL读取的不用转换
means.reverse()
stds.reverse()

print('mean = {}'.format(means))
print('std = {}'.format(stds))


'''
常用数据集的mean, std

CIFAR-10
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

CIFAR-100
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

Mini-ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

ImageNet
mean = [0.485, 0.456, 0.406],
std = [0.229, 0.224, 0.225])

COCO
mean = [0.471, 0.448, 0.408]
std = [0.234, 0.239, 0.242]


自定义数据集mean, std

HandGesture
mean = [0.6744646, 0.64552003, 0.61589843]
std = [0.1419178, 0.15874207, 0.18635085]

ImageWoof
mean = [0.48759258, 0.45700872, 0.39527148]
std = [0.25697905, 0.2497953, 0.25853014]
'''