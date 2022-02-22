from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)  # np.unique去除数组中的重复数字，并进行排序之后输出
    unique_tensor = torch.from_numpy(unique_np)
    # 复制数据
    tensor_res = tensor.new(unique_tensor.shape)  # 构建[相同数据类型]的新Tensor
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    # 获取bounding box的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 获取相交矩形的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # 交集区域
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # 并集区域
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)  # 416//13=32
    # feature map每条边格子的数量，416//32=13
    grid_size = inp_dim // stride
    # 一个方框属性个数，等于5+类别数量
    bbox_attrs = 5 + num_classes
    # anchors数量
    num_anchors = len(anchors)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)  # view是reshape功能，-1表示自适应
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, : 2] += x_y_offset  # bx=sigmoid(tx)+cx, by=sigmoid(ty)+cy

    # Log空间变换高度和宽度
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:, :, :4] *= stride
    
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    box_corner = prediction.new(prediction.shape)

    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)  # x1 = Cx - w/2
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)  # y1 = Cy - h/2
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)  # x2 = Cx + w/2
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)  # y2 = Cy + h/2
    prediction[:, :, :4] = box_corner[:, :, :4]  # 计算后的新坐标复制回去

    batch_size = prediction.size(0)  # 第0个维度是batch_size
    write = False  # 拼接结果到output中最后返回

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:, 5: 5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)

        image_pred = torch.cat(seq, 1)  # shape=(10647, 5+1+1=7)
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))  # torch.nonzero返回的是索引，会让non_zero_ind是个2维tensor
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:, -1])

        # 按照类别执行 NMS
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]  # 根据排序后的索引对应出的bbox的坐标与分数，依然为4x7的tensor
            idx = image_pred_class.size(0)  # detections的个数

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])

                except ValueError:
                    break

                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output

    except:
        return 0

def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim  # inp_dim是需要resize的尺寸（如416*416）
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names