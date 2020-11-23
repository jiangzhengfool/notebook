\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[www.cnblogs.com\](https://www.cnblogs.com/wj-1314/p/10422159.html)

　　在机器学习模型中，需要人工选择的参数称为超参数。比如随机森林中决策树的个数，人工神经网络模型中隐藏层层数和每层的节点个数，正则项中常数大小等等，他们都需要事先指定。超参数选择不恰当，就会出现欠拟合或者过拟合的问题。而在选择超参数的时候，有两个途径，一个是凭经验微调，另一个就是选择不同大小的参数，带入模型中，挑选表现最好的参数。

　　微调的一种方法是手工调制超参数，直到找到一个好的超参数组合，这么做的话会非常冗长，你也可能没有时间探索多种组合，所以可以使用 Scikit-Learn 的 GridSearchCV 来做这项搜索工作。下面让我们一一探索。

### 1，为什么叫网格搜索（GridSearchCV）？

　　GridSearchCV 的名字其实可以拆分为两部分，GridSearch 和 CV，即网格搜索和交叉验证。这两个名字都非常好理解。网格搜索，搜索的是参数，即在指定的参数范围内，按步长依次调整参数，利用调整的参数训练学习器，从所有的参数中找到在验证集上精度最高的参数，这其实是一个训练和比较的过程。

　　GridSearchCV 可以保证在指定的参数范围内找到精度最高的参数，但是这也是网格搜索的缺陷所在，他要求遍历所有可能参数的组合，在面对大数据集和多参数的情况下，非常耗时。

### 2，什么是 Grid Search 网格搜索？

　　Grid Search：一种调参手段；穷举搜索：在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果。其原理就像是在数组里找到最大值。这种方法的主要缺点是比较耗时！

　　所以网格搜索适用于三四个（或者更少）的超参数（当超参数的数量增长时，网格搜索的计算复杂度会呈现指数增长，这时候则使用随机搜索），用户列出一个较小的超参数值域，这些超参数至于的笛卡尔积（排列组合）为一组组超参数。网格搜索算法使用每组超参数训练模型并挑选验证集误差最小的超参数组合。

#### 2.1，以随机森林为例说明 GridSearch 网格搜索

　　下面代码，我们要搜索两种网格，一种是 n\_estimators，一种是 max\_features。GridSearch 会挑选出最适合的超参数值。

```
from sklearn.model\_selection import GridSearchCV
param\_grid = \[
{'n\_estimators': \[3, 10, 30\], 'max\_features': \[2, 4, 6, 8\]},
{'bootstrap': \[False\], 'n\_estimators': \[3, 10\], 'max\_features': \[2, 3, 4\]},
\]
 
forest\_reg = RandomForestRegressor()
grid\_search = GridSearchCV(forest\_reg, param\_grid, cv=5,
                          scoring='neg\_mean\_squared\_error')
 
grid\_search.fit(housing\_prepared, housing\_labels)
```

　　sklearn 根据 param\_grid 的值，首先会评估 3×4=12 种 n\_estimators 和 max\_features 的组合方式，接下来在会在 bootstrap=False 的情况下（默认该值为 True），评估 2×3=6 种 12 种 n\_estimators 和 max\_features 的组合方式，所以最终会有 12+6=18 种不同的超参数组合方式, 而每一种组合方式要在训练集上训练 5 次， 所以一共要训练 18×5=90 次，当训练结束后，你可以通过 best\_params\_获得最好的组合方式。

```
grid\_search.best\_params\_
```

　　输出结果如下：

```
{‘max\_features’: 8, ‘n\_estimators’: 30}
```

　　得到最好的模型：

```
grid\_search.best\_estimator\_
```

　　输出如下：

```
RandomForestRegressor(bootstrap=True, criterion=‘mse’, max\_depth=None,
max\_features=8, max\_leaf\_nodes=None, min\_impurity\_decrease=0.0,
min\_impurity\_split=None, min\_samples\_leaf=1,
min\_samples\_split=2, min\_weight\_fraction\_leaf=0.0,
n\_estimators=30, n\_jobs=1, oob\_score=False, random\_state=None,
verbose=0, warm\_start=False)
```

　　如果 GridSearchCV 初始化时，refit=True（默认的初始化值）, 在交叉验证时，一旦发现最好的模型（estimator）, 将会在整个训练集上重新训练，这通常是一个好主意，因为使用更多的数据集会提升模型的性能。

　　以上面有两个参数的模型为例，参数 a 有 3 中可能，参数 b 有 4 种可能，把所有可能性列出来，可以表示成一个 3\*4 的表格，其中每个 cell 就是一个网格，循环过程就像是在每个网格里遍历，搜索，所以叫 grid search。

#### 2.2，以 Xgboost 为例说明 GridSearch 网格搜索

　　下面以阿里 IJCAI 广告推荐数据集与 XgboostClassifier 分类器为例，用代码形式说明 sklearn 中 GridSearchCV 的使用方法。（此小例的代码是参考这里：[请点击我](https://blog.csdn.net/juezhanangle/article/details/80051256)）

```
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid\_search import GridSearchCV
  
  
#导入训练数据
traindata = pd.read\_csv("/traindata\_4\_3.txt",sep = ',')
traindata = traindata.set\_index('instance\_id')
trainlabel = traindata\['is\_trade'\]
del traindata\['is\_trade'\]
print(traindata.shape,trainlabel.shape)
  
  
#分类器使用 xgboost
clf1 = xgb.XGBClassifier()
  
#设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param\_dist = {
        'n\_estimators':range(80,200,4),
        'max\_depth':range(2,15,1),
        'learning\_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample\_bytree':np.linspace(0.5,0.98,10),
        'min\_child\_weight':range(1,9,1)
        }
 
 
#GridSearchCV参数说明，clf1设置训练的学习器
#param\_dist字典类型，放入参数搜索范围
#scoring = 'neg\_log\_loss'，精度评价方式设定为“neg\_log\_loss“
#n\_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
#n\_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
grid = GridSearchCV(clf1,param\_dist,cv = 3,scoring = 'neg\_log\_loss',n\_iter=300,n\_jobs = -1)
  
#在训练集上训练
grid.fit(traindata.values,np.ravel(trainlabel.values))
#返回最优的训练器
best\_estimator = grid.best\_estimator\_
print(best\_estimator)
#输出最优训练器的精度
```

　　这里关于网格搜索的几个参数在说明一下，评分参数 “scoring”，需要根据实际的评价标准设定，阿里的 IJCAI 的标准时 “neg\_log\_loss”，所以这里设定为 “neg\_log\_loss”，sklearn 中备选的评价标准如下：在一些情况下，sklearn 中没有现成的评价函数，sklearn 是允许我们自定义的，但是需要注意格式。

　　接下来看一下我们定义的评价函数：

```
import numpy as np
from sklearn.metrics import make\_scorer
  
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act\*sp.log(pred) + sp.subtract(1, act)\*sp.log(sp.subtract(1, pred)))
    ll = ll \* -1.0/len(act)
    return ll
  
#这里的greater\_is\_better参数决定了自定义的评价指标是越大越好还是越小越好
loss  = make\_scorer(logloss, greater\_is\_better=False)
score = make\_scorer(logloss, greater\_is\_better=True)
```

　　定义好以后，再将其带入 GridSearchCV 函数就好。

　　这里再贴一下常用的集成学习算法比较重要的需要调参的参数：

![](https://img2018.cnblogs.com/blog/1226410/201904/1226410-20190430112339421-889015410.png)

#### 2.3，以 SVR 为例说明 GridSearch 网格搜索

　　以两个参数的调优过程为例：

```
from sklearn.datasets import load\_iris
from sklearn.svm import SVC
from sklearn.model\_selection import train\_test\_split
 
iris\_data = load\_iris()
X\_train,X\_test,y\_train,y\_test = train\_test\_split(iris\_data.data,iris\_data.target,random\_state=0)
 
# grid search start
best\_score = 0
for gamma in \[0.001,0.01,1,10,100\]:
    for c in \[0.001,0.01,1,10,100\]:
        # 对于每种参数可能的组合，进行一次训练
        svm = SVC(gamma=gamma,C=c)
        svm.fit(X\_train,y\_train)
        score = svm.score(X\_test,y\_test)
        # 找到表现最好的参数
        if score > best\_score:
            best\_score = score
            best\_parameters = {'gamma':gamma,"C":c}
 
print('Best socre:{:.2f}'.format(best\_score))
print('Best parameters:{}'.format(best\_parameters))
```

　　输出结果：

```
Best socre:0.97
Best parameters:{'gamma': 0.001, 'C': 100}
```

#### 2.4  上面调参存在的问题是什么呢？

　　原始数据集划分成训练集和测试集以后，其中测试集除了用作调整参数，也用来测量模型的好坏；这样做导致最终的评分结果比实际效果好。（因为测试集在调参过程中，送到了模型里，而我们的目的是将训练模型应用到 unseen data 上）。

#### 2.5  解决方法是什么呢？

　　对训练集再进行一次划分，分为训练集和验证集，这样划分的结果就是：原始数据划分为 3 份，分别为：训练集，验证集和测试集；其中训练集用来模型训练，验证集用来调整参数，而测试集用来衡量模型表现好坏。

![](https://img2018.cnblogs.com/blog/1226410/201902/1226410-20190223134944232-1298995594.png)

　　代码：

```
from sklearn.datasets import load\_iris
from sklearn.svm import SVC
from sklearn.model\_selection import train\_test\_split
 
iris\_data = load\_iris()
# X\_train,X\_test,y\_train,y\_test = train\_test\_split(iris\_data.data,iris\_data.target,random\_state=0)
X\_trainval,X\_test,y\_trainval,y\_test = train\_test\_split(iris\_data.data,iris\_data.target,random\_state=0)
X\_train ,X\_val,y\_train,y\_val = train\_test\_split(X\_trainval,y\_trainval,random\_state=1)
# grid search start
best\_score = 0
for gamma in \[0.001,0.01,1,10,100\]:
    for c in \[0.001,0.01,1,10,100\]:
        # 对于每种参数可能的组合，进行一次训练
        svm = SVC(gamma=gamma,C=c)
        svm.fit(X\_train,y\_train)
        score = svm.score(X\_val,y\_val)
        # 找到表现最好的参数
        if score > best\_score:
            best\_score = score
            best\_parameters = {'gamma':gamma,"C":c}
 
# 使用最佳参数，构建新的模型
svm = SVC(\*\*best\_parameters)
 
# 使用训练集和验证集进行训练 more data always resultd in good performance
svm.fit(X\_trainval,y\_trainval)
 
# evalyation 模型评估
test\_score = svm.score(X\_test,y\_test)
 
print('Best socre:{:.2f}'.format(best\_score))
print('Best parameters:{}'.format(best\_parameters))
print('Best score on test set:{:.2f}'.format(test\_score))
```

　　结果：

```
Best socre:0.96
Best parameters:{'gamma': 0.001, 'C': 10}
Best score on test set:0.92
```

　　然而，这种简洁的 grid search 方法，其最终的表现好坏与初始数据的划分结果有很大的关系，为了处理这种情况，我们采用交叉验证的方式来减少偶然性。

#### 2.6，交叉验证改进 SVM 代码（Grid Search with Cross Validation）

 　　代码：

```
from sklearn.datasets import load\_iris
from sklearn.svm import SVC
from sklearn.model\_selection import train\_test\_split,cross\_val\_score
 
iris\_data = load\_iris()
# X\_train,X\_test,y\_train,y\_test = train\_test\_split(iris\_data.data,iris\_data.target,random\_state=0)
X\_trainval,X\_test,y\_trainval,y\_test = train\_test\_split(iris\_data.data,iris\_data.target,random\_state=0)
X\_train ,X\_val,y\_train,y\_val = train\_test\_split(X\_trainval,y\_trainval,random\_state=1)
# grid search start
best\_score = 0
for gamma in \[0.001,0.01,1,10,100\]:
    for c in \[0.001,0.01,1,10,100\]:
        # 对于每种参数可能的组合，进行一次训练
        svm = SVC(gamma=gamma,C=c)
        # 5 折交叉验证
        scores = cross\_val\_score(svm,X\_trainval,y\_trainval,cv=5)
        score = scores.mean()
        # 找到表现最好的参数
        if score > best\_score:
            best\_score = score
            best\_parameters = {'gamma':gamma,"C":c}
 
# 使用最佳参数，构建新的模型
svm = SVC(\*\*best\_parameters)
 
# 使用训练集和验证集进行训练 more data always resultd in good performance
svm.fit(X\_trainval,y\_trainval)
 
# evalyation 模型评估
test\_score = svm.score(X\_test,y\_test)
 
print('Best socre:{:.2f}'.format(best\_score))
print('Best parameters:{}'.format(best\_parameters))
print('Best score on test set:{:.2f}'.format(test\_score))
```

　　结果：

```
Best socre:0.97
Best parameters:{'gamma': 0.01, 'C': 100}
Best score on test set:0.97
```

　　交叉验证经常与网络搜索进行结合，作为参数评价的一种方法，这种方法叫做 grid search with cross validation。

　　sklearn 因此设计了一个这样的类 GridSearchCV，这个类实现 fit，predict，score 等方法。被当做一个 estimator，使用 fit 方法，该过程中：

*   （1） 搜索到最佳参数
*   （2）实例化了一个最佳参数的 estimator

### 3，RandomizedSearchCV——（随机搜索）

　　文献地址可以参考：请点击我

　　所谓的模型配置，一般统称为模型的超参数（Hyperparameters），比如 KNN 算法中的 K 值，SVM 中不同的核函数（Kernal）等。多数情况下，超参数等选择是无限的。在有限的时间内，除了可以验证人工预设几种超参数组合以外，也可以通过启发式的搜索方法对超参数组合进行调优。称这种启发式的超参数搜索方法为网格搜索。 

　　我们在搜索超参数的时候，如果超参数个数较少（三四个或者更少），那么我们可以采用网格搜索，一种穷尽式的搜索方法。但是当超参数个数比较多的时候，我们仍然采用网格搜索，那么搜索所需时间将会指数级上升。

 　　所以有人就提出了随机搜索的方法，随机在超参数空间中搜索几十几百个点，其中就有可能有比较小的值。这种做法比上面稀疏化网格的做法快，而且实验证明，随机搜索法结果比稀疏网格法稍好。

　　RandomizedSearchCV 使用方法和类 GridSearchCV 很相似，但他不是尝试所有可能的组合，而是通过选择每一个超参数的一个随机值的特定数量的随机组合，这个方法有两个优点：

*   如果你让随机搜索运行， 比如 1000 次，它会探索每个超参数的 1000 个不同的值（而不是像网格搜索那样，只搜索每个超参数的几个值）
*   你可以方便的通过设定搜索次数，控制超参数搜索的计算量。

　　RandomizedSearchCV 的使用方法其实是和 GridSearchCV 一致的，但它以随机在参数空间中采样的方式代替了 GridSearchCV 对于参数的网格搜索，在对于有连续变量的参数时，RandomizedSearchCV 会将其当做一个分布进行采样进行这是网格搜索做不到的，它的搜索能力取决于设定的 n\_iter 参数，同样的给出代码。

![](https://img2018.cnblogs.com/blog/1226410/201904/1226410-20190430100650150-385702207.png)

 　　代码如下：

```
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.grid\_search import RandomizedSearchCV
  
  
#导入训练数据
traindata = pd.read\_csv("/traindata.txt",sep = ',')
traindata = traindata.set\_index('instance\_id')
trainlabel = traindata\['is\_trade'\]
del traindata\['is\_trade'\]
print(traindata.shape,trainlabel.shape)
  
  
#分类器使用 xgboost
clf1 = xgb.XGBClassifier()
  
#设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param\_dist = {
        'n\_estimators':range(80,200,4),
        'max\_depth':range(2,15,1),
        'learning\_rate':np.linspace(0.01,2,20),
        'subsample':np.linspace(0.7,0.9,20),
        'colsample\_bytree':np.linspace(0.5,0.98,10),
        'min\_child\_weight':range(1,9,1)
        }
  
#RandomizedSearchCV参数说明，clf1设置训练的学习器
#param\_dist字典类型，放入参数搜索范围
#scoring = 'neg\_log\_loss'，精度评价方式设定为“neg\_log\_loss“
#n\_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
#n\_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
grid = RandomizedSearchCV(clf1,param\_dist,cv = 3,scoring = 'neg\_log\_loss',n\_iter=300,n\_jobs = -1)
  
#在训练集上训练
grid.fit(traindata.values,np.ravel(trainlabel.values))
#返回最优的训练器
best\_estimator = grid.best\_estimator\_
print(best\_estimator)
#输出最优训练器的精度
print(grid.best\_score\_)
```

　　建议使用随机搜索。

####  超参数搜索——网格搜索 & 并行搜索代码

```
#-\*- coding:utf-8 -\*-
 
#1.使用单线程对文本分类的朴素贝叶斯模型的超参数组合执行网格搜索
 
from sklearn.datasets import fetch\_20newsgroups
import numpy as np 
news = fetch\_20newsgroups(subset='all')
from sklearn.cross\_validation import train\_test\_split
#取前3000条新闻文本进行数据分割
X\_train,X\_test,y\_train,y\_test=train\_test\_split(news.data\[:3000\],
                                            news.target\[:3000\],test\_size=0.25,random\_state=33)
 
 
from sklearn.svm import SVC
from sklearn.feature\_extraction.text import TfidfVectorizer
#\*\*\*\*\*\*\*\*\*\*\*\*\*导入pipeline\*\*\*\*\*\*\*\*\*\*\*\*\*
from sklearn.pipeline import Pipeline
#使用Pipeline简化系统搭建流程，sklean提供的pipeline来将多个学习器组成流水线，通常流水线的形式为： 
#将数据标准化的学习器---特征提取的学习器---执行预测的学习器 
#将文本特征与分类器模型串联起来,\[(),()\]里有两个参数
#参数1:执行 vect = TfidfVectorizer(stop\_words='english',analyzer='word')操作
#参数2:执行 svc = SVC()操作
clf = Pipeline(\[('vect',TfidfVectorizer(stop\_words='english',analyzer='word')),('svc',SVC())\])
 
#这里需要试验的2个超参数svc\_gamma和svc\_C的元素个数分别为4、3,这样我们一共有12种超参数对集合
#numpy.linspace用于创建等差数列，numpy.logspace用于创建等比数列
#logspace中，开始点和结束点是10的幂
#例如logspace(-2,1,4)表示起始数字为10^-2，结尾数字为10^1即10，元素个数为4的等比数列
#parameters变量里面的key都有一个前缀,这个前缀其实就是在Pipeline中定义的操作名。二者相结合，使我们的代码变得十分简洁。
#还有注意的是，这里对参数名是<两条>下划线 \_\_
parameters = {'svc\_\_gamma':np.logspace(-2,1,4),'svc\_\_C':np.logspace(-1,1,3)}
 
#从sklearn.grid\_search中导入网格搜索模块GridSearchCV
from sklearn.grid\_search import GridSearchCV
#GridSearchCV参数解释:
#1.estimator : estimator(评估) object.
#2.param\_grid : dict or list of dictionaries
#3.verbose:Controls the verbosity(冗余度): the higher, the more messages.
#4.refit:default=True, Refit(再次拟合)the best estimator with the entire dataset
#5.cv : int, cross-validation generator 此处表示3折交叉验证
gs = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3)
 
#执行单线程网格搜索
gs.fit(X\_train,y\_train)
 
print gs.best\_params\_,gs.best\_score\_
 
#最后输出最佳模型在测试集上的准确性
print 'the accuracy of best model in test set is',gs.score(X\_test,y\_test) 
 
#小结：
#1.由输出结果可知，使用单线程的网格搜索技术 对朴素贝叶斯模型在文本分类任务中的超参数组合进行调优，
#  共有12组超参数组合\*3折交叉验证 =36项独立运行的计算任务
#2.在本机上，该过程一共运行了2.9min，寻找到最佳的超参数组合在测试集上达到的分类准确性为82.27%
```

```
#2.使用多线程对文本分类的朴素贝叶斯模型的超参数组合执行网格搜索
 
#n\_jobs=-1,表示使用该计算机的全部cpu
gs = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3,n\_jobs=-1)
gs.fit(X\_train,y\_train)
print gs.best\_params\_,gs.best\_score\_
#输出最佳模型在测试集上的准确性
print 'the accuracy of best model in test set is',gs.score(X\_test,y\_test) 
 
#小结：
#总任务相同的情况下，使用并行搜索技术进行计算的话，执行时间只花费了1.1min；
#而且最终所得的的best\_params\_和score没有发生变化，说明并行搜索可以在不影响准确性的前提下，
#有效的利用计算机的CPU资源，大大节省了最佳超参数的搜索时间。
```

### 4， 超参数估计的随机搜索和网格搜索的比较

　　**使用的数据集是小数据集 手写数字数据集 load\_digits() 分类   数据规模 5620\*64**

　　（sklearn 中的小数据可以直接使用，大数据集在第一次使用的时候会自动下载）

　　比较随机森林超参数优化的随机搜索和网格搜索。所有影响学习的参数都是同时搜索的（除了估计值的数量，它会造成时间 / 质量的权衡）。

　　随机搜索和网格搜索探索的是完全相同的参数空间。参数设置的结果非常相似，而随机搜索的运行时间要低的多。

　　随机搜索的性能稍差，不过这很可能是噪声效应，不会延续到外置测试集

　　注意：在实践中，人们不会使用网格搜索同时搜索这么多不同的参数，而是只选择那些被认为最重要的参数。

　　代码如下：

```
#\_\*\_coding:utf-8\_\*\_
# 输出文件开头注释的内容  \_\_doc\_\_的作用
'''
Python有个特性叫做文档字符串，即DocString ，这个特性可以让你的程序文档更加清晰易懂
'''
print(\_\_doc\_\_)
import numpy as np
from time import time
from scipy.stats import randint as sp\_randint
 
from sklearn.model\_selection import GridSearchCV
from sklearn.model\_selection import RandomizedSearchCV
from sklearn.datasets import load\_digits
from sklearn.ensemble import RandomForestClassifier
 
# get some  data
digits = load\_digits()
X, y = digits.data , digits.target
 
# build a classifier
clf = RandomForestClassifier(n\_estimators=20)
 
# utility function to report best scores
def report(results, n\_top= 3):
    for i in range(1, n\_top + 1):
        candidates = np.flatnonzero(results\['rank\_test\_score'\] == i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score : {0:.3f} (std: {1:.3f})".
                  format(results\['mean\_test\_score'\]\[candidate\],
                         results\['std\_test\_score'\]\[candidate\]))
            print("Parameters: {0}".format(results\['params'\]\[candidate\]))
            print("")
 
# 指定取样的参数和分布 specify parameters and distributions to sample from
param\_dist = {"max\_depth":\[3,None\],
              "max\_features":sp\_randint(1,11),
              "min\_samples\_split":sp\_randint(2,11),
              "bootstrap":\[True, False\],
              "criterion":\["gini","entropy"\]
              }
 
# run randomized search
n\_iter\_search = 20
random\_search = RandomizedSearchCV(clf,param\_distributions=param\_dist,
                                   n\_iter=n\_iter\_search,cv =5)
start = time()
random\_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n\_iter\_search))
report(random\_search.cv\_results\_)
 
# use a full grid over all parameters
param\_grid = {"max\_depth":\[3,None\],
              "max\_features":\[1, 3, 10\],
              "min\_samples\_split":\[2, 3, 10\],
              "bootstrap":\[True, False\],
              "criterion":\["gini","entropy"\]
    }
# run grid search
grid\_search = GridSearchCV(clf, param\_grid=param\_grid, cv =5)
start = time()
grid\_search.fit(X , y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid\_search.cv\_results\_\['params'\])))
report(grid\_search.cv\_results\_)
```

　　结果如下：

```
RandomizedSearchCV took 6.20 seconds for 20 candidates parameter settings.
Model with rank:1
Mean validation score : 0.930 (std: 0.031)
Parameters: {'bootstrap': False, 'criterion': 'entropy', 'max\_depth': None, 'max\_features': 6, 'min\_samples\_split': 5}
 
Model with rank:2
Mean validation score : 0.929 (std: 0.024)
Parameters: {'bootstrap': False, 'criterion': 'entropy', 'max\_depth': None, 'max\_features': 6, 'min\_samples\_split': 9}
 
Model with rank:3
Mean validation score : 0.924 (std: 0.020)
Parameters: {'bootstrap': False, 'criterion': 'gini', 'max\_depth': None, 'max\_features': 3, 'min\_samples\_split': 6}
 
 
Model with rank:1
Mean validation score : 0.932 (std: 0.023)
Parameters: {'bootstrap': False, 'criterion': 'gini', 'max\_depth': None, 'max\_features': 10, 'min\_samples\_split': 3}
 
Model with rank:2
Mean validation score : 0.931 (std: 0.014)
Parameters: {'bootstrap': False, 'criterion': 'gini', 'max\_depth': None, 'max\_features': 3, 'min\_samples\_split': 3}
 
Model with rank:3
Mean validation score : 0.929 (std: 0.021)
Parameters: {'bootstrap': False, 'criterion': 'entropy', 'max\_depth': None, 'max\_features': 3, 'min\_samples\_split': 2}
```

scikit-learn GridSearch 库概述
---------------------------

sklearn 的 Grid Search 官网地址：[请点击我](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)

### 1，GridSearchCV 简介

　　GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，很难得到结果。这个时候就需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调参，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可能会跳到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试，后续可以再拿 bagging 再优化。

　　通常算法不够好，需要调试参数时必不可少。比如 SVM 的惩罚因子 C，核函数 kernel，gamma 参数等，对于不同的数据使用不同的参数，结果效果可能差 1~5 个点，sklearn 为我们专门调试参数的函数 grid\_search。

### 2，GridSearchCV 参数说明

　　参数如下：

```
class sklearn.model\_selection.GridSearchCV(estimator, param\_grid, scoring=None, 
fit\_params=None, n\_jobs=None, iid=’warn’, refit=True, cv=’warn’, verbose=0, 
pre\_dispatch=‘2\*n\_jobs’, error\_score=’raise-deprecating’, return\_train\_score=’warn’)
```

　　说明如下：

　　　　1）estimator：选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个 scoring 参数，或者 score 方法：如 estimator = RandomForestClassifier(min\_sample\_split=100,min\_samples\_leaf = 20,max\_depth = 8,max\_features = 'sqrt' , random\_state =10),

　　　　2）param\_grid：需要最优化的参数的取值，值为字典或者列表，例如：param\_grid = param\_test1,param\_test1 = {'n\_estimators' : range(10,71,10)}

　　　　3）scoring = None ：模型评价标准，默认为 None，这时需要使用 score 函数；或者如 scoring = 'roc\_auc'，根据所选模型不同，评价准则不同，字符串（函数名），或是可调用对象，需要其函数签名，形如：scorer(estimator，X，y）；如果是 None，则使用 estimator 的误差估计函数。

　　　　4）fit\_para,s = None

　　　　5）n\_jobs = 1 ： n\_jobs：并行数，int：个数，-1：跟 CPU 核数一致，1：默认值

　　　　6）iid = True：iid：默认为 True，为 True 时，默认为各个样本 fold 概率分布一致，误差估计为所有样本之和，而非各个 fold 的平均。

　　　　7）refit = True ：默认为 True，程序将会以交叉验证训练集得到的最佳参数，重新对所有可能的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次 fit 一遍全部数据集。

　　　　8）cv = None：交叉验证参数，默认 None，使用三折交叉验证。指定 fold 数量，默认为 3，也可以是 yield 训练 / 测试数据的生成器。

　　　　9）verbose = 0 ,scoring = None　　verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

　　　　10）pre\_dispatch = '2\*n\_jobs' ：指定总共发的并行任务数，当 n\_jobs 大于 1 时候，数据将在每个运行点进行复制，这可能导致 OOM，而设置 pre\_dispatch 参数，则可以预先划分总共的 job 数量，使数据最多被复制 pre\_dispatch 次。

### 3，进行预测的常用方法和属性

*   grid.fit()  ：运行网格搜索
*   grid\_scores\_   ：给出不同参数情况下的评价结果
*   best\_params\_  ：描述了已取得最佳结果的参数的组合
*   best\_score\_    ：提供优化过程期间观察到的最好的评分
*   cv\_results\_  ：具体用法模型不同参数下交叉验证的结果

### 4，GridSearchCV 属性说明

　　　　（1） cv\_results\_ : dict of numpy (masked) ndarrays

　　　　具有键作为列标题和值作为列的 dict，可以导入到 DataFrame 中。注意，“params” 键用于存储所有参数候选项的参数设置列表。

　　　　（2） best\_estimator\_ : estimator

　　　　通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器。如果 refit = False，则不可用。

　　　　（3）best\_score\_ ：float  best\_estimator 的分数

　　　　（4）best\_parmas\_ : dict  在保存数据上给出最佳结果的参数设置

　　　　（5） best\_index\_ : int 对应于最佳候选参数设置的索引（cv\_results\_数组）

　　　　search.cv\_results \_ \['params'\] \[search.best\_index\_\] 中的 dict 给出了最佳模型的参数设置，给出了最高的平均分数（search.best\_score\_）。

　　　　（6）scorer\_ : function

　　　　Scorer function used on the held out data to choose the best parameters for the model.

　　　　（7）n\_splits\_ : int

　　　　The number of cross-validation splits (folds/iterations).  

### 3，利用决策树预测乳腺癌的例子（网格搜索算法优化）

#### 3.1 网格搜索算法与 K 折交叉验证理论知识

　　网格搜索算法是一种通过遍历给定的参数组合来优化模型表现的方法。

　　以决策树为例，当我们确定了要使用决策树算法的时候，为了能够更好地拟合和预测，我们需要调整它的参数。在决策树算法中，我们通常选择的参数是决策树的最大深度。

　　于是下面我们会给出一系列的最大深度的值，比如 {‘max\_depth’：\[1,2,3,4,5\] }，我们就会尽可能包含最优最大深度。

　　不过我们如何知道哪个最大深度的模型是最好的呢？我们需要一种可靠的评分方法，对每个最大深度的决策树模型都进行评价，这其中非常经典的一种方法就是交叉验证，下面我们就以 K 折交叉验证为例，详细介绍一下其算法过程。

　　首先我们先看一下数据集时如何分割的，我们拿到的原始数据集首先会按照一定的比例划分出训练集和测试集。比如下图，以 8:2 分割的数据集：

![](https://img2018.cnblogs.com/blog/1226410/201902/1226410-20190225095153176-1862814495.png)

　　训练集是用来训练我们的模型，它的作用就像我们平时做的练习题；测试集用来评估我们训练好的模型表现如何，它不能被提前被模型看到。

　　因此，在 K 折交叉验证中，我们用到的数据是训练集中的所有数据，我们将训练集的所有数据平均划分出 K 份（通常选择 K=10），取第 K 份作为验证集，它的作用就像我们用来估计高考分数的模拟题，余下的 K-1 份作为交叉验证的训练集。

　　对于我们最开始选择的决策树的 5 个最大深度，以 max\_depth=1 为例，我们先用第 2-10 份数据作为训练集训练模型，用第一份数据作为验证集对这次训练的模型进行评分，得到第一个分数；然后重新构建一个 max\_depth = 1 的决策树，用第 1 和 3-10 份数据作为训练集训练模型，用第 2 份数据作为验证集对这次训练的模型进行评分，得到第二个分数..... 以此类推，最后构建一个 max\_depth = 1 的决策树用第 1-9 份数据作为训练集训练模型，用第 10 份数据作为验证集对这次训练的模型进行评分，得到 10 个验证分数，然后计算着 10 个验证分数的平均分数，就是 max\_depth = 1 的决策树模型的最终验证分数。

![](https://img2018.cnblogs.com/blog/1226410/201902/1226410-20190225101032541-1168063435.png)

　　对于 max\_depth = 2,3,4,5 时，分别进行和 max\_depth =1 相同的交叉验证过程，得到他们的最终验证分数，然后我们就可以对这 5 个最大深度的决策树的最终验证分数进行比较，分数最高的那个就是最优最大深度，我们利用最优参数在全部训练集上训练一个新的模型，整个模型就是最优模型。

#### 3.2 简单的利用决策树预测乳腺癌的例子

代码：

```
from sklearn.model\_selection import GridSearchCV,KFold,train\_test\_split
from sklearn.metrics import make\_scorer , accuracy\_score
from sklearn.tree  import DecisionTreeClassifier
from sklearn.datasets import load\_breast\_cancer
import warnings
from sklearn.neighbors import KNeighborsClassifier as KNN
 
warnings.filterwarnings('ignore')
 
# load  data
data = load\_breast\_cancer()
print(data.data.shape)
print(data.target.shape)
# (569, 30)
# (569,)
X,y = data\['data'\] , data\['target'\]
 
X\_train,X\_test,y\_train,y\_test = train\_test\_split(
    X,y,train\_size=0.8 , random\_state=0
)
 
regressor = DecisionTreeClassifier(random\_state=0)
parameters = {'max\_depth':range(1,6)}
scorin\_fnc = make\_scorer(accuracy\_score)
kflod = KFold(n\_splits=10)
 
grid = GridSearchCV(regressor,parameters,scorin\_fnc,cv=kflod)
grid = grid.fit(X\_train,y\_train)
reg = grid.best\_estimator\_
 
print('best score:%f'%grid.best\_score\_)
print('best parameters:')
for key in parameters.keys():
    print('%s:%d'%(key,reg.get\_params()\[key\]))
 
print('test score : %f'%reg.score(X\_test,y\_test))
 
# import pandas as pd
# pd.DataFrame(grid.cv\_results\_).T
 
# 引入KNN训练方法
knn = KNN()
# 进行填充测试数据进行训练
knn.fit(X\_train,y\_train)
params = knn.get\_params()
score  = knn.score(X\_test,y\_test)
print("KNN 预测得分为：%s"%score)
```

　　结果：

```
(569, 30)
(569,)
best score:0.938462
best parameters:
max\_depth:4
test score : 0.956140
KNN 预测得分为：0.9385964912280702
```

### 问题一：AttributeError: 'GridSearchCV' object has no attribute 'grid\_scores\_'

#### 问题描述：

　　Python 运行代码的时候，到 gsearch1.grid\_scores\_ 时报错：

```
AttributeError: 'GridSearchCV' object has no attribute 'grid\_scores\_'
```

#### 原因：

　　之所以出现以上问题，原因在于 grid\_scores\_在 sklearn0.20 版本中已被删除，取而代之的是 cv\_results\_。

#### 解决方法：

　　将下面代码：

```
a,b,c = gsearch1.grid\_scores\_, gsearch1.best\_params\_, gsearch1.best\_score\_
```

　　换成：

```
a,b,c = gsearch1.cv\_results\_, gsearch1.best\_params\_, gsearch1.best\_score\_
```

### 问题二：ModuleNotFoundError: No module named 'sklearn.grid\_search'

#### 问题描述：

　　Python 运行代码时候，到 from  sklearn.grid\_search import GridSearchCV 时报错：

```
ModuleNotFoundError: No module named 'sklearn.grid\_search'
```

#### 原因：

　　sklearn.grid\_search 模块在 0.18 版本中被弃用，它所支持的类转移到 model\_selection 模板中。还要注意，新的 CV 迭代器的接口与这个模块的接口不同，sklearn.grid\_search 在 0.20 中被删除。

#### 解决方法：

　　将下面代码

```
from sklearn.grid\_search import GridSearchCV
```

　　修改成：

```
from sklearn.model\_selection import GridSearchCV
```

参考文献：https://blog.51cto.com/emily18/2088128

https://blog.csdn.net/jh1137921986/article/details/79827945

https://blog.csdn.net/juezhanangle/article/details/80051256