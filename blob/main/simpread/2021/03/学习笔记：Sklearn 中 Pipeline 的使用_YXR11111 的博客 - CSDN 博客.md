> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/qq_41598851/article/details/80957893)

学习笔记：通过学习很多大佬的博客然后综合总结，有些用的是大佬的原代码加上自己的理解

**1.Pipeline 的作用：**

        Pipeline 可以将许多算法模型串联起来，可以用于把多个 estamitors 级联成一个 estamitor, 比如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流。Pipleline 中最后一个之外的所有 estimators 都必须是变换器（transformers），最后一个 estimator 可以是任意类型（transformer，classifier，regresser）, 如果最后一个 estimator 是个分类器，则整个 pipeline 就可以作为分类器使用，如果最后一个 estimator 是个聚类器，则整个 pipeline 就可以作为聚类器使用。  

        主要带来两点好处：

        1. 直接调用 fit 和 predict 方法来对 pipeline 中的所有算法模型进行训练和预测。

        2. 可以结合 grid search 对参数进行选择.

**2. 串行化用法：**

**(1) 通过 steps 参数，设定数据处理流程。格式为 ('key','value')，key 是自己为这一 step 设定的名称，value 是对应的处理类。最后通过 list 将这些 step 传入。前 n-1 个 step 中的类都必须有 transform 函数，最后一步可有可无，一般最后一步为模型。使用最简单的 iris 数据集来举例：**

**in[ ]:**

```
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris=load_iris()
pipe=Pipeline(steps=[('pca',PCA()),('svc',SVC())])
pipe.fit(iris.data,iris.target)
```

**out[ ]:**

```
Pipeline(memory=None,
     steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('svc', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
```

**训练得到的是一个模型，可直接用来预测，预测时，数据会从 step1 开始进行转换，避免了模型用来预测的数据还要额外写代码实现。还可通过 pipe.score(X,Y) 得到这个模型在 X 训练集上的正确率：  
**

**in[ ]:**

```
pipe.predict(iris.data)

```

**out[ ]:**

```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

**(2) 通过 make_pipeline 函数实现：它是 Pipeline 类的简单实现，只需传入每个 step 的类实例即可，不需自己命名，自动将类的小写设为该 step 的名:**

**in[ ]:**

```
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler #用来解决离群点
make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
```

**out[ ]:**

```
Pipeline(memory=None,
     steps=[('robustscaler', RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)), ('lasso', Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=False))])
```

**同时可以通过 set_params 重新设置每个类里边需传入的参数，设置方法为 step 的 name__parma 名 = 参数值:**

**in [ ]:**

```
p=make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
p.set_params(lasso__alpha=0.0001) #将alpha从0.0005变成0.0001
```

**out[ ]:**

```
Pipeline(memory=None,
     steps=[('robustscaler', RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)), ('lasso', Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=False))])
```

并行化用法等学习到之后再更新啦~~~~~~~~