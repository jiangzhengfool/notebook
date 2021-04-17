> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/wonner_/article/details/103511168)

近年来机器学习和深度学习算法被越来越广泛的应用于解决对未知数据的预测问题。由于超参数的选择对模型最终的效果可能有极大的影响，为了使模型达到更好的效果，通常会面临超参数调优问题。但如何选择合适的超参数并没有一个明确的指导，并且同一模型面对随时间变化的数据，其超参数的选择都可能需要随着数据变化进行调节，更何况是原本就分布不同的两份数据。因此，人工指定超参数通常是根据经验或随机的方法进行尝试，深度学习里的 “调参工程师”，“炼丹” 等说法因此而得名。

既然调参是一个复杂并且耗费时间的工作（由于运行一次模型可能需要耗费一天甚至更长时间），有没有什么自动调参算法可以解放人力并且达到比经验调参更好的效果呢？已经有许多学者在自动化机器学习方向上进行研究，不仅包括超参数自动优化，还有自动化特征工程、自动化算法（模型）选择、自动化的神经体系结构搜索等。目前较常用的自动调参算法有 Grid Search(网格调参) 和 Bayesian Optimization(贝叶斯优化)。网格调参的思路很简单，给要调节的超参数一些选择，如果超参数调节范围为一个连续值则需要将其离散化（如使用等间距采样）。之后遍历所有的超参数组合找到这些组合中最优的方案。然而实际上这些组合中可能不包含全局最优解，并且当要调节的超参数比较多时，产生的组合数也会特别多，为每个组合跑一遍模型验证需要花费大量的时间。对于 XGBoost 这种可调节超参数很多的模型，为了平衡时间开销和优化结果，通常会进行分组调参，即采用控制变量的方式，先调整部分超参数，将调出的最优超参固定，继续调节还没有调节的超参。在我的上篇博客[竞赛常用集成学习框架 Boosting 算法总结 (XGBoost、LightGBM)(附代码)](https://blog.csdn.net/wonner_/article/details/103130113) 中给出了这种超参调节的代码。这种局部优化的方式可能距离全局最优解更远，但为了平衡时间我们不得不这样做。

我们发现，Grid Search 对每种超参数组合都是独立计算的，当次计算不会利用之间计算得到的信息，这就造成了计算的浪费。而贝叶斯优化算法则利用了之间轮计算的信息对之后的超参选择给出指导，基于过去的结果搜索未知参数范围，走向可能是全局最优的方向。贝叶斯优化可以基于不同的代理模型，分为以下三类，并给出基于这三类算法实现的 python 库：

*   TPE(Tree Parzen Estimator, 树形 Parzen 评估器)：Hyperopt, Optuna
*   SMAC(随机森林回归)：SMAC3
*   GP(高斯过程)：**scikit-optimize**, GPyOpt, Botorch, spearmint, fmfn/BayesianOptimization

以上总结是 Jeff Dean 在 ICML 2019 上关于 AutoML 的演讲上给出的，原文链接：[An Overview of AutoML Libraries Used in Industry](https://towardsdatascience.com/overview-of-automl-from-pycon-jp-2019-c8996954692f)。综合考虑到质量和速度，在贝叶斯优化上他推荐使用的库是 scikit-optimize。因此下文我们会给出基于高斯过程的贝叶斯优化算法的原理，并在最后给出使用 scikit-optimize 库对 XGBoost 和 LightGBM 的超参数进行贝叶斯优化的代码。

我们的优化目标是使机器学习模型的验证误差最小，也就是说，需要找到自变量为超参数和因变量为验证误差的函数最小值点。为了下文方便说明（其实是懒得画图），我们改为找函数的最大值点，其思想是一致的。贝叶斯优化根据前几轮超参数组合计算出的真实的验证误差，通过高斯过程，得到在超参数所有取值上验证误差的期望均值和方差。均值越大代表该组超参数的最终期望效果越好，方差越大表示这组超参数的效果不确定性越大。因此均值大或方差大对应的那组超参数是我们下一步想要带入模型计算验证其效果的。

![](https://img-blog.csdnimg.cn/20191213211726456.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvbm5lcl8=,size_16,color_FFFFFF,t_70)

t=2 表示已经通过模型计算出了两组超参数的验证误差，即为图中的黑点。黑色实线为假设已知的验证误差随超参数变化函数，需找到其最大值。黑色虚线代表期望均值，紫色区域代表期望方差。那么下一组超参数应该如何选择呢？因为前面提到均值代表期望的最终结果，当然是越大越好，但我们不能每次都挑选均值最大的，因为有的点方差很大也有可能存在全局最优解，因此选择均值大的点我们称为 exploritation（开发），选择方差大的点我们称为 exploration（探索）。均值和方差比例通过定义 acquisition function 确定，对开发和探索进行权衡。简单的 acquisition function 有 Upper condence bound 算法：

![](https://private.codecogs.com/gif.latex?x_%7Bt%7D%3D%5Carg%20%5Cmax%20_%7Bx%20%5Cin%20%5Cmathcal%7BX%7D%7D%20%5Calpha_%7Bt%7D%28x%29%3D%5Carg%20%5Cmax%20_%7Bx%20%5Cin%20%5Cmathcal%7BX%7D%7D%20%5Cmu_%7Bt-1%7D%28x%29&plus;%5Cbeta_%7Bt%7D%5E%7B1%20/%202%7D%20%5Csigma_%7Bt-1%7D%28x%29)

计算均值和方差的加权和，其中![](https://private.codecogs.com/gif.latex?%5Cbeta_%7Bt%7D)的值是根据理论分析推出来的，随时间递增；在实际应用里面，为了简便也可直接把![](https://private.codecogs.com/gif.latex?%5Cbeta_%7Bt%7D)设成一个常数。除此之外还有很多复杂的 acquisition function，可参考博客[贝叶斯优化 / Bayesian Optimization](https://zhuanlan.zhihu.com/p/76269142)。

根据应用我们选择一个合适的 acquisition function，求得其最大值，对应的这组超参数值就是贝叶斯优化算法根据之前的计算结果推荐的下一组计算的超参数值。如上图中的绿色曲线，选择其最大值点继续训练模型。

![](https://img-blog.csdnimg.cn/20191214102738396.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvbm5lcl8=,size_16,color_FFFFFF,t_70)

然后我们将新的计算结果加入到历史结果中，继续通过高斯过程计算均值方差，通过 acquisition function 计算下一组带入训练模型的超参数值。不断重复上述步骤拟合最终的曲线，找出一组最好的值。虽然机器学习模型的超参数不一定是这么完美的曲线，但可以从概率上找到一个较好的参数。

下文将给出使用 scikit-optimize 对 XGBoost 和 LightGBM 进行超参数调优的代码。

使用 pip 安装库：

```
pip install scikit-optimize

```

在 anaconda 上安装库，运行以下任意一个即可：

```
conda install -c conda-forge scikit-optimize
conda install -c conda-forge/label/gcc7 scikit-optimize
conda install -c conda-forge/label/cf201901 scikit-optimize
```

在 BayesSearchCV 类中实现了贝叶斯优化，但下载的库中这个类存在问题，会报错 TypeError: __init__() got an unexpected keyword argument 'fit_params'，因此自定义 FixedBayesSearchCV 修复这个问题。以多分类问题为例。

```
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
 
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
 
class FixedBayesSearchCV(BayesSearchCV):
    """
    Dirty hack to avoid compatibility issues with sklearn 0.2 and skopt.
    Credit: https://www.kaggle.com/c/home-credit-default-risk/discussion/64004
    For context, on why the workaround see:
        - https://github.com/scikit-optimize/scikit-optimize/issues/718
        - https://github.com/scikit-optimize/scikit-optimize/issues/762
    """
    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                n_points=1, iid=True, refit=True, cv=None, verbose=0,
                pre_dispatch='2*n_jobs', random_state=None,
                error_score='raise', return_train_score=False):
        """
        See: https://github.com/scikit-optimize/scikit-optimize/issues/762#issuecomment-493689266
        """
 
        # Bug fix: Added this line
        self.fit_params = fit_params
 
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
 
        # Removed the passing of fit_params to the parent class.
        super(BayesSearchCV, self).__init__(
                estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
                refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                error_score=error_score, return_train_score=return_train_score)
 
    def _run_search(self, x):
        raise BaseException('Use newer skopt')
 
model_lgb = lgb.LGBMClassifier(
            learning_rate=0.1,   # 学习率
            n_estimators=10000,    # 树的个数
            max_depth=10,         # 树的最大深度
            num_leaves=31,        # 叶子节点个数 'leaf-wise'
            min_split_gain=0,     # 节点分裂所需的最小损失函数下降值
            objective='multiclass', # 多分类
            metric='multiclass',  # 评价函数
            num_class=4,          # 多分类问题类别数
            subsample=0.8,        # 样本随机采样作为训练集的比例
            colsample_bytree=0.8, # 使用特征比例
            seed=1)
 
# 若包含类别变量，将其类型设置为category，astype('category')
# lightgbm scikit-optimize
def lgb_auto_para_tuning_bayesian(model_lgb,X,Y):
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.80, random_state=0)
    # cv：交叉验证 n_points：并行采样的超参组数
    opt = FixedBayesSearchCV(model_lgb,cv=3,n_points=2,n_jobs=4,verbose=1,
        search_spaces={
            'learning_rate': Real(0.008, 0.01),
            'max_depth': Integer(3, 10),
            'num_leaves': Integer(31, 127),
            'min_split_gain':Real(0.0,0.4),
            'min_child_weight':Real(0.001,0.002),
            'min_child_samples':Integer(18,22),
            'subsample':Real(0.6,1.0),
            'subsample_freq':Integer(3,5),
            'colsample_bytree':Real(0.6,1.0),
            'reg_alpha':Real(0,0.5),
            'reg_lambda':Real(0,0.5)
        },
         fit_params={
                 'eval_set':[(test_x, test_y)],
                 'eval_metric': 'multiclass',
                 'early_stopping_rounds': 50
                 })
    opt.fit(train_x,train_y)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(test_x, test_y))
    print("Best parameters: ", opt.best_params_)
    print("Best estimator:", opt.best_estimator_)
 
model_xgb = XGBClassifier(
            learning_rate =0.01,  # 学习率
            n_estimators=10000,   # 树的个数
            max_depth=6,         # 树的最大深度
            min_child_weight=1,  # 叶子节点样本权重加和最小值sum(H)
            gamma=0,             # 节点分裂所需的最小损失函数下降值
            subsample=0.8,       # 样本随机采样作为训练集的比例
            colsample_bytree=0.8, # 使用特征比例
            objective= 'multi:softmax', # 损失函数(这里为多分类）
            num_class=4,         # 多分类问题类别数
            scale_pos_weight=1,  # 类别样本不平衡
            seed=1)
 
# xgboost scikit-optimize
def xgb_auto_para_tuning_bayesian(model_xgb,X,Y):
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.80, random_state=0)
    opt = FixedBayesSearchCV(model_xgb,cv=3,n_points=2,n_jobs=4,verbose=1,
        search_spaces={
            'learning_rate': Real(0.008, 0.01),
            'max_depth': Integer(3, 10),
            'gamma':Real(0,0.5),
            'min_child_weight':Integer(1,8),
            'subsample':Real(0.6,1.0),
            'colsample_bytree':Real(0.6,1.0),
            'reg_alpha':Real(0,0.5),
            'reg_lambda':Real(0,0.5)
        },
         fit_params={
                 'eval_set': [(test_x, test_y)],
                 'eval_metric': 'mlogloss',
                 'early_stopping_rounds': 50
                 })
    opt.fit(train_x,y=train_y)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(test_x, test_y))
    print("Best parameters: ", opt.best_params_)
    print("Best estimator:", opt.best_estimator_)
```

scikit-optimize 官方文档：[https://scikit-optimize.github.io/#skopt.BayesSearchCV](https://scikit-optimize.github.io/#skopt.BayesSearchCV)