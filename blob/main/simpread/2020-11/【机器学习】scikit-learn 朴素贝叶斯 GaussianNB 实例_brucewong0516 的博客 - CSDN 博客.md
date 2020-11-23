\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[blog.csdn.net\](https://blog.csdn.net/brucewong0516/article/details/78798359)

1、什么是朴素贝叶斯

在所有的机器学习分类算法中，朴素贝叶斯和其他绝大多数的分类算法都不同。对于大多数的分类算法，比如决策树, KNN, 逻辑回归，支持向量机等，他们都是判别方法，也就是直接学习出特征输出 Y 和特征 X 之间的关系，要么是决策函数 Y=f(X), 要么是条件分布 P(Y|X)。但是朴素贝叶斯却是生成方法，也就是直接找出特征输出 Y 和特征 X 的联合分布 P(X,Y), 然后用 P(Y|X)=P(X,Y)/P(X) 得出。朴素贝叶斯很直观，计算量也不大，在很多领域有广泛的应用。

首先回顾一下朴素贝叶斯相关的统计学知识：

贝叶斯学派的思想可以概括为先验概率 + 数据 = 后验概率。也就是说我们在实际问题中需要得到的后验概率，可以通过先验概率和数据一起综合得到。

我们先看看条件独立公式，如果 X 和 Y 相互独立，则有：_P(A,B)=P(A)P(B)_

我们接着看看条件概率公式：P(AB)=P(A|B)P(B)=P(B|A)P(A)

接着看看全概率公式:![](https://img-blog.csdn.net/20171214004415631?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYnJ1Y2V3b25nMDUxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

从上面的公式很容易得出贝叶斯公式：![](https://img-blog.csdn.net/20171214004426719?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYnJ1Y2V3b25nMDUxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

2、scikit-learn 朴素贝叶斯类库

朴素贝叶斯是一类比较简单的算法，scikit-learn 中朴素贝叶斯类库的使用也比较简单。相对于决策树，KNN 之类的算法，朴素贝叶斯需要关注的参数是比较少的，这样也比较容易掌握。在 scikit-learn 中，一共有 3 个朴素贝叶斯的分类算法类。分别是 GaussianNB，MultinomialNB 和 BernoulliNB。其中 GaussianNB 就是先验为高斯分布的朴素贝叶斯，MultinomialNB 就是先验为多项式分布的朴素贝叶斯，而 BernoulliNB 就是先验为伯努利分布的朴素贝叶斯。

这三个类适用的分类场景各不相同，主要根据数据类型来进行模型的选择。一般来说，如果样本特征的分布大部分是连续值，使用 GaussianNB 会比较好。如果如果样本特征的分大部分是多元离散值，使用 MultinomialNB 比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用 BernoulliNB。

a>>>GaussianNB Naive\_Bayes  
GaussianNB 类的主要参数仅有一个，即先验概率 priors，对应 Y 的各个类别的先验概率 P(Y=Ck)。这个值默认不给出，如果不给出此时 P(Y=Ck)=mk/m。其中 m 为训练集样本总数量，mk 为输出为第 k 类别的训练集样本数。如果给出的话就以 priors 为准。 在使用 GaussianNB 的 fit 方法拟合数据后，我们可以进行预测。此时预测有三种方法，包括 predict，predict\_log\_proba 和 predict\_proba。

predict 方法就是我们最常用的预测方法，直接给出测试集的预测类别输出。

predict\_proba 则不同，它会给出测试集样本在各个类别上预测的概率。容易理解，predict\_proba 预测出的各个类别概率里的最大值对应的类别，也就是 predict 方法得到类别。

predict\_log\_proba 和 predict\_proba 类似，它会给出测试集样本在各个类别上预测的概率的一个对数转化。转化后 predict\_log\_proba 预测出的各个类别对数概率里的最大值对应的类别，也就是 predict 方法得到类别。

下面给一个具体的例子：

```
import numpy as np
X = np.array(\[\[-1, -1\], \[-2, -1\], \[-3, -2\], \[1, 1\], \[2, 1\], \[3, 2\]\])
Y = np.array(\[1, 1, 1, 2, 2, 2\])
from sklearn.naive\_bayes import GaussianNB
clf = GaussianNB()
#拟合数据
clf.fit(X, Y)
print("==Predict result by predict==")
print(clf.predict(\[\[-0.8, -1\]\]))
print("==Predict result by predict\_proba==")
print(clf.predict\_proba(\[\[-0.8, -1\]\]))
print("==Predict result by predict\_log\_proba==")
print(clf.predict\_log\_proba(\[\[-0.8, -1\]\]))
```

此外，GaussianNB 一个重要的功能是有 partial\_fit 方法，这个方法的一般用在如果训练集数据量非常大，一次不能全部载入内存的时候。这时我们可以把训练集分成若干等分，重复调用 partial\_fit 来一步步的学习训练集，非常方便。后面讲到的 MultinomialNB 和 BernoulliNB 也有类似的功能。

b>>>MultinomialNB Naive\_Bayes

MultinomialNB 参数比 GaussianNB 多，但是一共也只有仅仅 3 个。其中，参数 alpha 即为上面的常数λ，如果你没有特别的需要，用默认的 1 即可。如果发现拟合的不好，需要调优时，可以选择稍大于 1 或者稍小于 1 的数。布尔参数 fit\_prior 表示是否要考虑先验概率，如果是 false, 则所有的样本类别输出都有相同的类别先验概率。否则可以自己用第三个参数 class\_prior 输入先验概率，或者不输入第三个参数 class\_prior 让 MultinomialNB 自己从训练集样本来计算先验概率，此时的先验概率为 P(Y=Ck)=mk/m。其中 m 为训练集样本总数量，mk 为输出为第 k 类别的训练集样本数。

![](https://img-blog.csdn.net/20171214004506929?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYnJ1Y2V3b25nMDUxNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

c>>>BernoulliNB Naive\_Bayes

BernoulliNB 一共有 4 个参数，其中 3 个参数的名字和意义和 MultinomialNB 完全相同。唯一增加的一个参数是 binarize。这个参数主要是用来帮 BernoulliNB 处理二项分布的，可以是数值或者不输入。如果不输入，则 BernoulliNB 认为每个数据特征都已经是二元的。否则的话，小于 binarize 的会归为一类，大于 binarize 的会归为另外一类。

在使用 BernoulliNB 的 fit 或者 partial\_fit 方法拟合数据后，我们可以进行预测。此时预测有三种方法，包括 predict，predict\_log\_proba 和 predict\_proba。由于方法和 GaussianNB 完全一样，这里就不累述了。

3、朴素贝叶斯的主要优缺点

朴素贝叶斯的主要优点有：

　　　　1）朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率。

　　　　2）对小规模的数据表现很好，能个处理多分类任务，适合增量式训练，尤其是数据量超出内存时，我们可以一批批的去增量训练。

　　　　3）对缺失数据不太敏感，算法也比较简单，常用于文本分类。

朴素贝叶斯的主要缺点有：　　　

　　　　1） 理论上，朴素贝叶斯模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为朴素贝叶斯模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，在属性个数比较多或者属性之间相关性较大时，分类效果不好。而在属性相关性较小时，朴素贝叶斯性能最为良好。对于这一点，有半朴素贝叶斯之类的算法通过考虑部分关联性适度改进。

　　　　2）需要知道先验概率，且先验概率很多时候取决于假设，假设的模型可以有很多种，因此在某些时候会由于假设的先验模型的原因导致预测效果不佳。

　　　　3）由于我们是通过先验和数据来决定后验的概率从而决定分类，所以分类决策存在一定的错误率。

　　　　4）对输入数据的表达形式很敏感。