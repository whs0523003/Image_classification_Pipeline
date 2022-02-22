from __future__ import division
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
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="目标检测结果置信度阈值", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS非极大值抑制阈值", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="配置文件", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重", default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="网络输入分辨率，分辨率越高,则准确率越高，速度越慢; 反之亦然", default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="待检测视频目录", default="video.avi", type=str)
    
    return parser.parse_args()
    
args = arg_parse()

batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

def write(x, results):
    c1 = tuple(x[1:3].int())  # c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int())  # c2为方框右下角坐标x2,y2
    img = results
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

# 检测阶段
videofile = args.videofile

cap = cv2.VideoCapture(videofile)

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 24
savedPath = './det/savevideo.avi'
ret, frame = cap.read()
videoWriter = cv2.VideoWriter(savedPath, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 最后为视频图片的形状

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        # 只进行前向计算，不计算梯度
        with torch.no_grad():
            output = model(Variable(img, volatile=True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(int(args.reso) / im_dim, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
        output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2
        # 将坐标映射回原始图片
        output[:, 1:5] /= scaling_factor
        # 将超过了原始图片范围的方框坐标限定在图片范围之内
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        # coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字
        classes = load_classes('data/coco.names')
        # 读入包含100个颜色的文件pallete，里面是100个三元组序列
        colors = pkl.load(open("pallete", "rb"))
        # 将每个方框的属性写在图片上
        list(map(lambda x: write(x, frame), output))

        cv2.imshow("frame", frame)

        videoWriter.write(frame)  # 每次循环，写入该帧

        key = cv2.waitKey(1)
        # 如果有按键输入则返回按键值编码，输入q返回113
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
    else:
        videoWriter.release()  # 结束循环的时候释放
        break