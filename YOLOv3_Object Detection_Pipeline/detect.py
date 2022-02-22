from __future__ import division

import sys
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest='images', help="待检测图像存放的目录", default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="检测结果保存目录，det保存检测结果的目录", default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="目标检测结果置信度阈值", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS非极大值抑制阈值", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="配置文件", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重", default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="网络输入分辨率，分辨率越高,则准确率越高，速度越慢; 反之亦然", default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="缩放尺度用于检测", default="1,2,3", type=str)
    return parser.parse_args()

args = arg_parse()

images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0

CUDA = torch.cuda.is_available()  # GPU环境是否可用
num_classes = 80  # coco 数据集有80类

classes = load_classes("data/coco.names")

# 初始化网络并载入权重
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

# 网络输入数据大小
model.net_info["height"] = args.reso  # model类中net_info是一个字典。height是图片的宽高，因为图片缩放到416x416，所以宽高一样大
inp_dim = int(model.net_info["height"])  # inp_dim是网络输入图片尺寸（如416*416）
assert inp_dim % 32 == 0  # 如果设定的输入图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim > 32

# 如果GPU可用, 模型切换到cuda中运行
if CUDA:
    model.cuda()

model.eval()

read_dir = time.time()

# 加载待检测图像列表
try:
    imlist = [osp.join(images, img) for img in os.listdir(images)]

except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))

except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# 存储结果目录
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()  # 开始载入图片的时间。load_batch - read_dir 得到读取所有图片路径的时间
loaded_ims = [cv2.imread(x) for x in imlist]  # 使用opencv加载图像，读入所有图片，一张图片的数组在loaded_ims列表中保存为一个元素

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)  # repeat(*size), 沿着指定维度复制数据，size维度必须和数据本身维度要一致

leftover = 0  # 创建 batch，将所有测试图片按照batch_size分成多个batch
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size, len(im_batches))])) for i in range(num_batches)]

write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()

# 开始计时，计算开始检测的时间。
start_det_loop = time.time()

for i, batch in enumerate(im_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # 取消梯度计算
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    if type(prediction) == int:
        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i*batch_size

    if not write:
        output = prediction  
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()

# 对所有的输入的检测结果
try:
    # 当检测到目标时，输出目标
    output
except NameError:
    # 当所有图片都有没检测到目标时，退出程序
    print("没有检测到任何目标")
    exit()

# 最后输出output_recast - start_det_loop计算的是从开始检测，到去掉低分，NMS操作的时间。
output_recast = time.time()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
scaling_factor = torch.min(int(args.reso)/im_dim_list, 1)[0].view(-1, 1)
output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim_list[:, 0].view(-1, 1))/2
output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim_list[:, 1].view(-1, 1))/2
output[:, 1:5] /= scaling_factor  # 缩放至原图大小尺寸

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

# 开始载入颜色文件的时间
class_load = time.time()
# 绘图
colors = pkl.load(open("../YOLOv3教程版，无训练/YOLO_v3_tutorial_from_scratch-master/pallete", "rb"))  # 读入包含100个颜色的文件pallete，里面是100个三元组序列

# 开始画方框的文字的时间
draw = time.time()

# x为映射到原始图片中一个方框的属性(ind,x1,y1,x2,y2,s,s_cls,index_cls)，results列表保存了所有测试图片，一个元素对应一张图片
def write(x, results):
    c1 = tuple(x[1:3].int())  # c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int())  # c2为方框右下角坐标x2,y2
    img = results[int(x[0])]  # 在results中找到x方框所对应的图片，x[0]为方框所在图片在所有测试图片中的序号
    cls = int(x[-1])
    color = random.choice(colors)  # 随机选择一个颜色，用于后面画方框的颜色
    label = "{0}".format(classes[cls])  # label为这个框所含目标类别名字的字符串
    cv2.rectangle(img, c1, c2, color, 1)  # 在图片上画出(x1,y1,x2,y2)矩形，即我们检测到的目标方框
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]  # 得到一个包含目标名字字符的方框的宽高
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4  # 得到包含目标名字的方框右下角坐标c2，这里在x,y方向上分别加了3、4个像素
    cv2.rectangle(img, c1, c2, color, -1)  # 在图片上画一个实心方框，我们将在方框内放置目标类别名字
    # 在图片上写文字，(c1[0], c1[1] + t_size[1] + 4)为字符串的左下角坐标
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
# 读取所有图片路径的时间
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
# 读入所有图片，并将图片按照batch size分成不同batch的时间
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
# 从开始检测到到去掉低分，NMS操作得到output的时间
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
# 这里output映射回原图的时间
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
# 画框和文字的时间
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
# 从开始载入图片到所有结果处理完成，平均每张图片所消耗时间
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()