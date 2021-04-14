> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/tsyccnh/article/details/79163834)

关于交叉熵在 loss 函数中使用的理解
====================

交叉熵（cross entropy）是深度学习中常用的一个概念，一般用来求目标与预测值之间的差距。以前做一些分类问题的时候，没有过多的注意，直接调用现成的库，用起来也比较方便。最近开始研究起对抗生成网络（GANs），用到了交叉熵，发现自己对交叉熵的理解有些模糊，不够深入。遂花了几天的时间从头梳理了一下相关知识点，才算透彻的理解了，特地记录下来，以便日后查阅。

信息论
---

交叉熵是信息论中的一个概念，要想了解交叉熵的本质，需要先从最基本的概念讲起。

### 1 信息量

首先是信息量。假设我们听到了两件事，分别如下：  
事件 A：巴西队进入了 2018 世界杯决赛圈。  
事件 B：中国队进入了 2018 世界杯决赛圈。  
仅凭直觉来说，显而易见事件 B 的信息量比事件 A 的信息量要大。究其原因，是因为事件 A 发生的概率很大，事件 B 发生的概率很小。所以当越不可能的事件发生了，我们获取到的信息量就越大。越可能发生的事件发生了，我们获取到的信息量就越小。那么信息量应该和事件发生的概率有关。

假设 $X$是一个离散型随机变量，其取值集合为$χ$, 概率分布函数 $p(x)=Pr(X=x),x∈χ$, 则定义事件 $X=x0$的信息量为：

$$I(x0)=−log(p(x0))$$

由于是概率所以

$p(x0)$

的取值范围是

$[0,1]$

, 绘制为图形如下：

  
![](https://img-blog.csdn.net/20180125164333234?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdHN5Y2NuaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  
可见该函数符合我们对信息量的直觉

### 2 熵

考虑另一个问题，对于某个事件，有 $n$种可能性，每一种可能性都有一个概率 $p(xi)$  
这样就可以计算出某一种可能性的信息量。举一个例子，假设你拿出了你的电脑，按下开关，会有三种可能性，下表列出了每一种可能的概率及其对应的信息量

<table><thead><tr><th align="center">序号</th><th align="center">事件</th><th align="center">概率 p</th><th align="center">信息量 I</th></tr></thead><tbody><tr><td align="center">A</td><td align="center">电脑正常开机</td><td align="center">0.7</td><td align="center">-log(p(A))=0.36</td></tr><tr><td align="center">B</td><td align="center">电脑无法开机</td><td align="center">0.2</td><td align="center">-log(p(B))=1.61</td></tr><tr><td align="center">C</td><td align="center">电脑爆炸了</td><td align="center">0.1</td><td align="center">-log(p(C))=2.30</td></tr></tbody></table>

> 注：文中的对数均为自然对数

我们现在有了信息量的定义，而熵用来表示所有信息量的期望，即：  

$$H(X)=−∑i=1np(xi)log(p(xi))$$

 

其中 n 代表所有的 n 种可能性，所以上面的问题结果就是  

$$H(X)=−[p(A)log(p(A))+p(B)log(p(B))+p(C))log(p(C))]=0.7×0.36+0.2×1.61+0.1×2.30=0.804$$

 

然而有一类比较特殊的问题，比如投掷硬币只有两种可能，字朝上或花朝上。买彩票只有两种可能，中奖或不中奖。我们称之为 0-1 分布问题（二项分布的特例），对于这类问题，熵的计算方法可以简化为如下算式：  

$$H(X)=−∑i=1np(xi)log(p(xi))=−p(x)log(p(x))−(1−p(x))log(1−p(x))$$

 

### 3 相对熵（KL 散度）

相对熵又称 KL 散度, 如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异

维基百科对相对熵的定义

> In the context of machine learning, DKL(P‖Q) is often called the information gain achieved if P is used instead of Q.

即如果用 P 来描述目标问题，而不是用 Q 来描述目标问题，得到的信息增量。

在机器学习中，P 往往用来表示样本的真实分布，比如 [1,0,0] 表示当前样本属于第一类。Q 用来表示模型所预测的分布，比如[0.7,0.2,0.1]  
直观的理解就是如果用 P 来描述样本，那么就非常完美。而用 Q 来描述样本，虽然可以大致描述，但是不是那么的完美，信息量不足，需要额外的一些 “信息增量” 才能达到和 P 一样完美的描述。如果我们的 Q 通过反复训练，也能完美的描述样本，那么就不再需要额外的“信息增量”，Q 等价于 P。

KL 散度的计算公式：  

$$(3.1)DKL(p||q)=∑i=1np(xi)log(p(xi)q(xi))$$

 

n 为事件的所有可能性。

$DKL$

的值越小，表示 q 分布和 p 分布越接近

### 4 交叉熵

对式 3.1 变形可以得到：  

$$DKL(p||q)=∑i=1np(xi)log(p(xi))−∑i=1np(xi)log(q(xi))=−H(p(x))+[−∑i=1np(xi)log(q(xi))]$$

 

等式的前一部分恰巧就是 p 的熵，等式的后一部分，就是交叉熵：  

$$H(p,q)=−∑i=1np(xi)log(q(xi))$$

 

在机器学习中，我们需要评估 label 和 predicts 之间的差距，使用 KL 散度刚刚好，即 $DKL(y||y^)$，由于 KL 散度中的前一部分$−H(y)$不变，故在优化过程中，只需要关注交叉熵就可以了。所以一般在机器学习中直接用用交叉熵做 loss，评估模型。

机器学习中交叉熵的应用
-----------

### 1 为什么要用交叉熵做 loss 函数？

在线性回归问题中，常常使用 MSE（Mean Squared Error）作为 loss 函数，比如：  

$$loss=12m∑i=1m(yi−yi^)2$$

这里的 m 表示 m 个样本的，loss 为 m 个样本的 loss 均值。  
MSE 在线性回归问题中比较好用，那么在逻辑分类问题中还是如此么？

### 2 交叉熵在单分类问题中的使用

这里的单类别是指，每一张图像样本只能有一个类别，比如只能是狗或只能是猫。  
交叉熵在单分类问题上基本是标配的方法  

$$(2.1)loss=−∑i=1nyilog(yi^)$$

上式为一张样本的 loss 计算方法。式 2.1 中 n 代表着 n 种类别。  
举例说明, 比如有如下样本  

  
![](https://img-blog.csdn.net/20180125164444783?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdHN5Y2NuaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

对应的标签和预测值

<table><thead><tr><th>*</th><th>猫</th><th>青蛙</th><th>老鼠</th></tr></thead><tbody><tr><td>Label</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Pred</td><td>0.3</td><td>0.6</td><td>0.1</td></tr></tbody></table>

那么  

$$loss=−(0×log(0.3)+1×log(0.6)+0×log(0.1)=−log(0.6)$$

 

对应一个 batch 的 loss 就是  

$$loss=−1m∑j=1m∑i=1nyjilog(yji^)$$

m 为当前 batch 的样本数

### 3 交叉熵在多分类问题中的使用

这里的多类别是指，每一张图像样本可以有多个类别，比如同时包含一只猫和一只狗  
和单分类问题的标签不同，多分类的标签是 n-hot。  
比如下面这张样本图，即有青蛙，又有老鼠，所以是一个多分类问题  

  
![](https://img-blog.csdn.net/20180125164456925?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdHN5Y2NuaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

对应的标签和预测值

<table><thead><tr><th>*</th><th>猫</th><th>青蛙</th><th>老鼠</th></tr></thead><tbody><tr><td>Label</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Pred</td><td>0.1</td><td>0.7</td><td>0.8</td></tr></tbody></table>

值得注意的是，这里的 Pred 不再是通过 softmax 计算的了，这里采用的是 sigmoid。将每一个节点的输出归一化到 [0,1] 之间。所有 Pred 值的和也不再为 1。换句话说，就是每一个 Label 都是独立分布的，相互之间没有影响。所以交叉熵在这里是单独对每一个节点进行计算，每一个节点只有两种可能值，所以是一个二项分布。前面说过对于二项分布这种特殊的分布，熵的计算可以进行简化。

同样的，交叉熵的计算也可以简化，即  

$$loss=−ylog(y^)−(1−y)log(1−y^)$$

 

注意，上式只是针对一个节点的计算公式。这一点一定要和单分类 loss 区分开来。  
例子中可以计算为：  

$$loss 猫 =−0×log(0.1)−(1−0)log(1−0.1)=−log(0.9)loss 蛙 =−1×log(0.7)−(1−1)log(1−0.7)=−log(0.7)loss 鼠 =−1×log(0.8)−(1−1)log(1−0.8)=−log(0.8)$$

 

单张样本的 loss 即为 $loss=loss 猫 +loss 蛙 +loss 鼠$  
每一个 batch 的 loss 就是：

$$loss=∑j=1m∑i=1n−yjilog(yji^)−(1−yji)log(1−yji^)$$

 

式中 m 为当前 batch 中的样本量，n 为类别数。

总结
--

路漫漫，要学的东西还有很多啊。

参考：

> [https://www.zhihu.com/question/65288314/answer/244557337](https://www.zhihu.com/question/65288314/answer/244557337)  
> [https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)  
> [https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)