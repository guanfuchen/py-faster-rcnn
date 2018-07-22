# face faster rcnn

将faster rcnn模型应用到人脸检测中，仅仅需要修改faster_rcnn_end2end/test.prototxt或者faster_rcnn_end2end/train.prototxt最后的cls_score为2（pascal_voc为21）和bbox_pred为8（pascal_voc为84）即可。

这里训练包括faster_rcnn_end2end和faster_rcnn_alt_opt这两种训练策略，这里一般使用faster_rcnn_end2end端到端训练策略。


---

下面步骤主要介绍CUSTOMDATABASE构造过程，比如INRIAPerson。

## 构建数据集CUSTOMDATABASE

```
INRIAPerson原始数据格式
|-- INRIAPerson/
    |-- 70X134H96/
    |-- 96X160H96/
    |-- Test/
    |-- test_64x128_H96/
    |-- Train/
    |-- train_64x128_H96/
```

```
其中Annotations放置标注文件，Images放置训练图像，ImageSets中train.txt放置训练图像文件名列表
INRIA_Person_devkit/
|-- data/
    |-- Annotations/
         |-- *.txt (Annotation files)
    |-- Images/
         |-- *.png (Image files)
    |-- ImageSets/
         |-- train.txt
```

## 增加CUSTOMDATABASE.py

在$PY_FASTER_RCNN/lib/datasets中增加[inria.py](https://github.com/deboc/py-faster-rcnn/blob/master/lib/datasets/inria.py)

## 更新factory.py

在$PY_FASTER_RCNNlib/datasets/factory.py中修改factory.py，比如INRIA Person。

```
from datasets.inria import inria
inria_devkit_path = '$PY_FASTER_RCNN/data/INRIA_Person_devkit'
for split in ['train', 'test']:
    name = '{}_{}'.format('inria', split)
    __sets[name] = (lambda split=split: inria(split, inria_devkit_path))
```


## 修改网络模型适应CUSTOMDATABASE

其中类别包括背景合计为C，对于Person vs Background那么C=2

```
修改num_classes为C
修改cls_score中的num_classes为C
修改bbox_pred中的num_classes为4*C
```


---
## 参考资料

- [Py-faster-rcnn实现自己的数据train和demo](https://blog.csdn.net/samylee/article/details/51201744) py faster rcnn实现自己的数据训练。
- [How to train Faster R-CNN on my own dataset ?](https://github.com/rbgirshick/py-faster-rcnn/issues/243)
- [Train Py-Faster-RCNN on Another Dataset](https://github.com/deboc/py-faster-rcnn/tree/master/help)
