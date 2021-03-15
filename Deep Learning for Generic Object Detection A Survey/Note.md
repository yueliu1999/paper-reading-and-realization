# Deep Learning for Generic Object Detection: A Survey



## Generic Object Detection

- 问题定义

  给定一张图片，决定图片中是否含有为预先定义的**类别**（predefined categories），如果包含，则需要返回每个实例的准确的**空间位置和范围**（spatial location and extent of each instance）

  - 类别

  - 空间位置和范围

    - a bounding box
    - pixel wise segmentation mask
    - closed boundary

    未来的挑战应该在于pixel level

- 主要的挑战

  理想的generic object detection需要是

  - high quality / accuracy
    - localization Acc
    - recognition Acc
  - high efficiency
    - time efficiency
    - memory efficiency
    - storage efficiency

  ![Snipaste_2021-03-14_08-27-31](..\Deep Learning for Generic Object Detection A Survey\screen_shot\Snipaste_2021-03-14_08-27-31.png)

- 近20年的发展

  ![Snipaste_2021-03-14_08-40-55](..\Deep Learning for Generic Object Detection A Survey\screen_shot\Snipaste_2021-03-14_08-40-55.png)

  准确的标注很难获取，必须考虑**克服标准的困难**，或者可以在**更小的数据集上学习**

  



## 深度学习的介绍



## 数据集和性能指标

- 数据集

  - PASCAL VOC

  - ImageNet

  - MS COCO

  - Open Image

  - ILSVRC Image Large Scale Visual Recongnition Challenge

    

- 性能指标

  - detection speed in Frames per Second(FPS)
  - precision
  - recall

  最常用的metric

  - Average precision(AP)

    一个测试图片I被预测的输出为

    {bbox, 类别, 置信度confidence}

    一个输出被认定为TP(True positive)需要的条件

    1. 类别预测正确
    2. 预测的bbox和真实的bbox的IOU不小于阈值，一般来说阈值为0.5

    不符合条件的则认定为FP(False positive)

    置信度

  - mean AP(m AP)

    所有物体类别的AP取平均

  

