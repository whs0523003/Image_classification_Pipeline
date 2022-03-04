'''
__init__.py文件的作用

该文件的作用就是相当于把自身整个文件夹（models）当作一个包来管理，每当有外部import的时候，就会自动执行里面的函数。

没有__init__.py，导入models里的各个网络的py文件是这样的：
from models import dnnnet, cnnnet, lenet, alexnet, mtnet, googlenet, resnet

有了__init__.py，且在以下定义from.import(...)后就可以直接：
from models import *
'''


from . import (
        alexnet,
        cnnnet,
        dnnnet,
        googlenet,
        lenet,
        mobilenetv1,
        mtnet,
        resnet,
        vggnet
)