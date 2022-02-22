##文件夹介绍：

datasets：读取数据集
- HandGesture的训练集和测试集需要分开读取，训练集是data/train/classes/img，测试集是data/test/img
    
HandGesture数据集需要

model：模型

train：训练模型
- CPU和GPU的选择：
  - 数据和网络需要在同一设备上才可以
  - 将网络转到指定device上是inplace操作，而将数据转到指定device上不是，需要再进行赋值操作
  - correct_test += (predicted == labels).squeeze().sum().to("cpu").numpy()，GPU上无法进行numpy数组类型的操作，需要先把数据放到CPU上才可以转成数组类型
  

predict：用训练好的模型进行预测

utilis：一些辅助函数

plot：根据metrics里保存的npy数据画图。train里也有相同部分，这里可以直接使用plot.py单独绘图

datasets：保存一些数据集特殊的解析方法

metrics：保存性能指标

checkpoints：保存模型

inputs：放入待预测数据

camera：摄像头

注：需要把datasets文件夹和models文件夹设置为source文件夹

##图像分类Pipeline功能介绍：
- 0.配置cuda
- 1.读取数据，数据预处理
- 2.模型
- 3.损失函数
- 4.优化器
- 5.训练，每个epoch打印训练集和测试集上的loss和acc
- 6.保存模型，每个epoch保存1次作为best，有更好的则替换，最终保存1个best和1个最后epoch的模型
- 7.保存指标，每个epoch的训练集和测试集上的loss和acc
- 8.画图，每个epoch的训练集和测试集上的loss和acc


- -计时，每个epoch耗时和总耗时
- -加载pytorch自带的训练好的模型
- -迁移学习
- -apex混合精度加速
- -tensorboard绘图（模型搭建，精确率，损失，学习率，权重分布，预测图片信息）。打开方式：anaconda对应环境（common）进入events.tf.tfevents所在目录的上级目录后输入tensorboard --logdir./激活tensorboard，然后把端口（localhost:xxxx/）复制到网址打开

