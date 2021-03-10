> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/minixuezhen/article/details/81516949?utm_medium=distribute.pc_relevant.none-task-blog-)

本地 python3 版本  
用 matplotlib 或 seaborn 作图时，出现以下告警，表明是中文显示的问题。

```
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

```

（此条可能只适用于 mac，上面这条不管用，改成下面的字体）

```
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

```

管用！