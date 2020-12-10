> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/SMF0504/article/details/71698354)

train loss 不断下降，test loss 不断下降，说明网络仍在学习;

train loss 不断下降，test loss 趋于不变，说明网络过拟合;

train loss 趋于不变，test loss 不断下降，说明数据集 100% 有问题;

train loss 趋于不变，test loss 趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;

train loss 不断上升，test loss 不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。