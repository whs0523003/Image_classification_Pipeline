from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import *

def get_test_input():
    img = cv2.imread("../YOLOv3教程版，无训练/dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))            # 把图片缩放至网络的输入尺寸416*416
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # opencv的BGR（h, w, c) -> pytorch的RGB(c, h, w)
    img_ = img_[np.newaxis, :, :, :] / 255.0     # 在第0位添加一个通道用于存放batch，并标准化
    img_ = torch.from_numpy(img_).float()        # 转成float类型
    img_ = Variable(img_)                        # 转成Variable类型
    return img_

def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')               # 读取每一行
    lines = [x for x in lines if len(x) > 0]      # 去掉空行
    lines = [x for x in lines if x[0] != '#']     # 去掉#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格

    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":                          # [表示cfg文件中一个层(块)的开始
            if len(block) != 0:                     # 如果块内已经存了信息, 说明是上一个块的信息还没有保存
                blocks.append(block)                # 那么这个块（字典）加入到blocks列表中去
                block = {}                          # 重置block字典
            block["type"] = line[1:-1].rstrip()     # 把cfg的[]中的块名作为键type的值
        else:
            key, value = line.split("=")            # 按等号分割
            block[key.rstrip()] = value.lstrip()    # 左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对

    blocks.append(block)  # 退出循环，将最后一个未加入的block加进去

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def create_modules(blocks):
    net_info = blocks[0]   # blocks[0]存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息
    module_list = nn.ModuleList()  # 用于存储每个block，每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    prev_filters = 3  # 初始值对应于输入数据3通道，用来存储需要持续追踪被应用卷积层的卷积核数量（上一层的卷积核数量（或特征图深度））
    output_filters = []  # 不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，将每个模块的输出卷积核数量添加到output_filters列表中
    
    for index, x in enumerate(blocks[1:]):  # 迭代block[1:] 而不是blocks，因为blocks的第一个元素是一个net块，它不属于前向传播
        module = nn.Sequential()  # 每个块用nn.sequential()创建为了一个module，一个module有多个层

        # 如果该模块是卷积层
        if (x["type"] == "convolutional"):
            # 获取激活函数/批归一化/卷积层参数（通过字典的键获取值）
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False  # 卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True  # 卷积层后无BN层就需要bias
        
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 开始创建并添加相应层
            # 添加卷积层
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            # 添加BN层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            # 检查激活层
            # 如果线性激活或leaky ReLU激活
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        
        # 如果是上采样层
        # 没有使用 Bilinear2dUpsampling，实际使用的为最近邻插值
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        # 如果是route层
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # 开始route
            start = int(x["layers"][0])
            # 如果已经存在则终止
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # 如果是正值
            if start > 0: 
                start = start - index

            # 若end>0，由于end=end-index，再执行index+end输出的还是第end层的特征
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # 若end<0，则end还是end，输出index+end(而end<0)故index向后退end层的特征
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]

            else:
                filters = output_filters[index + start]
    
        # 对应跳连接的shortcut层
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()  # 使用空层，因为它还要执行一个非常简单的操作（加）。没必要更新filters变量，因为它只是将前一层的特征图添加到后面的层上而已。
            module.add_module("shortcut_{}".format(index), shortcut)
            
        # 如果是YOLO层
        # YOLO层是检测层
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)  # 调用parse_cfg函数
        self.net_info, self.module_list = create_modules(self.blocks)  # 调用create_modules函数
        
    def forward(self, x, CUDA):
        modules = self.blocks[1:]  # 除了net块之外的所有，forward这里用的是blocks列表中的各个block块字典
        outputs = {}   # 缓存route层的输出
        
        write = 0  # write表示是否遇到第一个检测。0表示收集器未初始化，1表示收集器已初始化，我们只需要将检测图与收集器级联起来即可
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                # 第二个元素同理
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)  # 第二个参数设为1，因为我们希望将特征图沿anchor数量的维度级联起来

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]  # 求和运算，它只是将前一层的特征图添加到后面的层上而已
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                # 从net_info(实际就是blocks[0]，即[net])中得到输入维度
                inp_dim = int(self.net_info["height"])

                # 得到类别数量
                num_classes = int(module["classes"])
                x = x.data  # 这里得到的是预测的yolo层feature map
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if not write:
                    detections = x  # detections的初始化在有预测值出来时才能进行
                    write = 1  # 用write=1标记，当后面的分数出来后，直接concatenate操作即可

                else:
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x
        
        return detections

    def load_weights(self, weightfile):
        # 打开权重文件
        fp = open(weightfile, "rb")

        header = np.fromfile(fp, dtype=np.int32, count=5)  # 这里读取前5个值的权重
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype=np.float32)  # 加载np.ndarray中的剩余权重，权重是以float32类型存储的
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]  # blocks中的第一个元素是网络参数和图像的描述，所以从blocks[1]开始读入

            # 如果模块是卷积层则读取权重，其他层则忽略
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])  # 当有bn层时，BN对应值为1
                except:
                    batch_normalize = 0
            
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # BN层的权重数量
                    num_bn_biases = bn.bias.numel()
        
                    # 加载权重
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 将加载的权重投射到模型权重的维度中
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    # 将从weights文件中得到的权重bn_biases复制到model中(bn.bias.data)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                # 如果BN的检查结果不是True，只需要加载卷积层的偏置项
                else:
                    # bias的数量
                    num_biases = conv.bias.numel()
                
                    # 加载权重
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # 根据模型权重的维度重塑已加载的权重
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    # 最后复制数据
                    conv.bias.data.copy_(conv_biases)

                # 加载卷积层的权值
                num_weights = conv.weight.numel()
                
                # 对权重做同上的处理
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)