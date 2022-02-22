# YOLO_v3目标检测教程

YOLO目标检测和图像分类工作不太一样，它的训练部分和预测部分是两个不同
的部分，实现的逻辑也比图像分类任务要复杂一些。

以下是效果图的样子：

![Detection Example](https://i.imgur.com/m2jwnen.png)

## 关于没有的train.py
该项目仅作为YOLOv3的教程部分，包含训练部分的完整部分见：
https://github.com/ayooshkathuria/pytorch-yolo-v3

## 关于imgs文件夹
该文件夹用于存放所有待测试图片

## 关于pallete
一个pickle文件，其中包含很多可以随机选择的颜色。用于画检测框