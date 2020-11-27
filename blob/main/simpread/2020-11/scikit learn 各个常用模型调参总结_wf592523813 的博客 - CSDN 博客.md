> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/wf592523813/article/details/86382037)

[SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
=============================================================================================

1.  对数据进行归一化 （simple scaling）
2.  使用 RBF kernel
3.  使用 cross_validation 和 grid_search 得到最佳参数 gamma 和 C
4.  使用得到的最优 C 和 gamma 训练训练数据
5.  测试

svm 的 C
-------

C 是惩罚系数，即对误差的宽容度。一般可以选择为：10^t , t=[- 4，4] 就是 0.0001 到 10000。c 越高，说明越不能容忍出现误差, 容易过拟合。C 越小，容易欠拟合。C 过大或过小，泛化能力变差

### 常用核函数

![](https://img-blog.csdnimg.cn/20190112221931373.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dmNTkyNTIzODEz,size_16,color_FFFFFF,t_70)

核函数参数设置
-------

*   **线性核函数**：没有专门需要设置的参数
*   **多项式核函数**：有三个参数。  
    **-d 用来设置多项式核函数的最高次项次数**，也就是公式中的 d，默认值是 3，一般选择 1-11：1 3 5 7 9 11，也可以选择 2,4，6…。  
    **-g 用来设置核函数中的 gamma 参数设置**，也就是公式中的 gamma，默认值是 1/k（特征数）。  
    **-r 用来设置核函数中的 coef0**，也就是公式中的第二个 r，默认值是 0。
*   **RBF 核函数**：有一个参数。**-g 用来设置核函数中的 gamma 参数设置**，也就是公式中 gamma，默认值是 1/k（k 是特征数）。
*   **sigmoid 核函数又叫做 S 形内核 **：有两个参数。  
    **-g 用来设置核函数中的 gamma 参数设置**，也就是公式中 gamma，默认值是 1/k（k 是特征数）。一般可选 1 2 3 4  
    **-r 用来设置核函数中的 coef0**，也就是公式中的第二个 r，默认值是 0。一般可选 0.2 0.4 0.6 0.8 1

rbf 核函数 gamma ：
---------------

gamma 是选择 RBF 函数作为 kernel 后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，**gamma 越大，支持向量越少，gamma 值越小，支持向量越多**。****支持向量的个数影响训练与预测的速度。****

需要注意的就是 gamma 的物理意义，RBF 的幅宽会影响每个支持向量对应的高斯的作用范围，从而影响泛化性能。如果 gamma 设的太大，方差会很小，方差很小的高斯分布长得又高又瘦， 会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，(如果让方差无穷小，则理论上，高斯核的 SVM 可以拟合任何非线性数据，但容易过拟合) 而测试准确率不高的可能，就是通常说的过训练；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率。

**使用 grid Search 调参比较简单**，详见[交叉验证与网格搜索算法](https://mp.csdn.net/mdeditor/86309547#)，而且看起来很 naïve。有两个优点：  
可以得到全局最优，(C,gamma) 相互独立，便于并行化进行。缺点：耗时！！！

[KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
===========================================================================================================================================

![](https://img-blog.csdnimg.cn/20190112222937808.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dmNTkyNTIzODEz,size_16,color_FFFFFF,t_70)

超参数为:**n_neighbors /weight/p/algorithm**（只有当 weight=distance 的时候，p 值才有意义）

*   **n_neighbors**：取邻近点的个数 k。k 取 1-9 测试
*   **weight**：距离的权重，uniform：一致的权重；distance：距离的倒数作为权重
*   **p**: 闵可斯基距离的 p 值; p=1: 即欧式距离；p=2: 即曼哈顿距离；p 取 1-6 测试
*   **algorithm**： ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’  
    **在实践中，选择较小的邻居个数（例如 3 或 5 个）效果较好**，  
    **sklearn 中默认使用欧氏距离构建 KNN 模型速度很快，若训练集很大（特征数多或样本数多），预测速度可能较慢.  
    对于稀疏数据集（大多数特征值为 0），KNN 效果很不好.**

[Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
=========================================================================================================================================

*   **alpha** : 先验平滑因子，默认等于 1，当等于 1 时表示拉普拉斯平滑。只有在伯努利模型和多项式模型中存在

[Decision Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
================================================================================================================================================

*   **特征选择标准 criterion**：使用 "**gini**"或者"**entropy**"，前者代表基尼系数，后者代表信息增益。一般说使用默认的基尼系数"gini"，即 CART 算法。除非要使用 ID3,C4.5 的最优特征选择方法。
*   **特征划分点选择标准 splitter**：使用 "**best**"或者"**random**"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"
*   **划分时考虑的最大特征数 max_features**：可以使用很多种类型的值，默认是 "**None**", 意味着划分时考虑所有的特征数；如果是"**log2**"意味着划分时最多考虑 log2N 个特征；如果是"sqrt"或者"auto" 意味着划分时最多考虑√N 个特征。如果是**整数**，代表考虑的特征绝对数。如果是**浮点数**，代表考虑特征百分比，即考虑（百分比 xN）取整后的特征数。其中 N 为样本总特征数。一般来说，如果样本特征数不多，比如小于 50，用默认的 "None" 就可以了，如果特征数非常多，可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
*   **决策树最大深 max_depth**：默认可以**不输入**，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。**常用的可以取值 10-100 之间。**
*   **内部节点再划分所需最小样本数 min_samples_split**：这个值限制了子树继续划分的条件，如果某节点的样本数少于 min_samples_split，则不会继续再尝试选择最优特征来进行划分。 ** 默认是 2.** 如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。若有大概 10 万样本，建立决策树时，可选择 min_samples_split=10 作为参考。
*   **叶子节点最少样本数 min_samples_leaf**：这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 **默认是 1**, 可以输入最少的样本数的**整数**，或者最少样本数占样本总数的**百分比**。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。之前的 10 万样本使用 min_samples_leaf 的值为 5，仅供参考。
*   **叶子节点最小的样本权重和 min_weight_fraction_leaf**：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 **默认是 0**，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重。
*   **最大叶子节点数 max_leaf_nodes**：通过限制最大叶子节点数，可以防止过拟合，**默认是 "None”**，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
*   **类别权重 class_weight**：指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。这里可以自己**指定各个样本的权重**，或者用 “**balanced**”，如果使用 “balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，**选择默认的 "None"**  
    **节点划分最小不纯度 min_impurity_split**：这个值限制了决策树的增长，如果某节点的不纯度 (基尼系数，信息增益，均方差，绝对差) 小于这个阈值，则该节点不再生成子节点。即为叶子节点 。  
    **数据是否预排序 presort**：**布尔值，默认是 False 不排序**。一般来说，如果样本量少或者限制了一个深度很小的决策树，设置为 true 可以让划分点选择更加快，决策树建立的更加快。如果样本量太大的话，反而没有什么好处。问题是样本量少的时候，速度本来就不慢。所以这个值一般懒得理它就可以了。  
    **最重要的是**：  
    **最大特征数 max_features，  
    最大深度 max_depth，  
    内部节点再划分所需最小样本数 min_samples_split  
    叶子节点最少样本数 min_samples_leaf**

[RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
======================================================================================================================================================

**1. RF 的 Bagging 框架的参数**：

1.  **n_estimators**: 弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说 n_estimators 太小，容易欠拟合，n_estimators 太大，计算量会太大，并且 n_estimators 到一定的数量后，再增大 n_estimators 获得的模型提升会很小，所以一般选择一个适中的数值。**默认是 100**。在实际调参的过程中，我们常常将 n_estimators 和 learning_rate 一起考虑。
    
2.  **oob_score**: 即是否采用袋外样本来评估模型的好坏。**默认 False**。个人**推荐设置为 True**，因为袋外分数反应了一个模型拟合后的泛化能力。
    
3.  **criterion:** 即 CART 树做划分时对特征的评价标准。分类模型和回归模型的损失函数是不一样的。分类 RF 对应的 CART 分类树**默认是基尼系数 gini**, 另一个可选择的标准是信息增益。回归 RF 对应的 CART 回归树默认是均方差 mse，另一个可以选择的标准是绝对值差 mae。一般来说选择默认的标准就已经很好的。
    
    **重要的参数是 n_estimators，即 RF 最大的决策树个数。**
    

**2.** **RF 决策树参数与决策树参数相同，参见上面的 DecisionTree**， **除去 splitter**，在结点进行分裂的时候，先随机取固定个特征，然后选择最好的分裂属性这种方式。

**scikit-learn 中实现了两种随机森林算法**，一种是 **RandomForest**，另外一种是 **ExtraTrees**。  
_ExtraTrees 在最好的几个（依然可以指定 sqrt 与 log2) 分裂属性中随机选择一个来进行分裂。_

[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
============================================================================================================================================================

*   **惩罚项 penalty**： **‘l1’ or ‘l2’**, **默认: ‘l2’** ，在调参时如果我们主要的目的只是为了解决过拟合，一般 penalty 选择 L2 正则化就够了。但是如果选择 L2 正则化发现还是过拟合，即预测效果差的时候，就可以考虑 L1 正则化。如果模型的特征非常多，希望一些不重要的特征系数归零，从而让模型系数稀疏化，也可以使用 L1 正则化。
    
*   **solver 优化方法**
    
    *   **liblinear**：使用了开源的 liblinear 库实现，内部使用了坐标轴下降法来迭代优化损失函数。
    *   **lbfgs**：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    *   **newton-cg**：牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    *   **sag**：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候，SAG 是一种线性收敛算法，这个速度远比 SGD 快。
*   **C**：正则化系数λ的倒数，float 类型，默认为 1.0。必须是正浮点型数。像 SVM 一样，越小的数值表示越强的正则化。
    
*   **class_weight**：用于标示分类模型中各种类型的权重，可以是一个字典或者’balanced’字符串，**默认为不输入**，也就是不考虑权重，即为 None。如果选择输入的话，可以选择 **balanced** 让类库自己计算类型权重，或者自己输入各个类型的权重。举个例子，比如对于 0,1 的二元模型，我们可以定义 **class_weight={0:0.9,1:0.1}**，这样类型 0 的权重为 90%，而类型 1 的权重为 10%。如果 class_weight 选择 balanced，那么类库会根据训练样本量来计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。当 class_weight 为 balanced 时，类权重计算方法如下：n_samples / (n_classes * np.bincount(y))。n_samples 为样本数，n_classes 为类别数量，np.bincount(y) 会输出每个类的样本数，例如 y=[1,0,0,1,1], 则 np.bincount(y)=[2,3]。
    
*   **max_iter**：算法收敛最大迭代次数，int 类型，**默认为 10**。仅在正则化优化算法为 newton-cg, sag 和 lbfgs 才有用，算法收敛的最大迭代次数。
    
*   **multi_class**：分类方式选择参数，str 类型，可选参数为 **ovr 和 multinomial**，**默认为 ovr**。ovr 即前面提到的 one-vs-rest(OvR)，而 multinomial 即前面提到的 many-vs-many(MvM)。如果是二元逻辑回归，ovr 和 multinomial 并没有任何区别，区别主要在多元逻辑回归上。如果选择了 ovr，则 4 种损失函数的优化方法 liblinear，newton-cg,lbfgs 和 sag 都可以选择。但是**如果选择了 multinomial, 则只能选择 newton-cg, lbfgs 和 sag 了。**
    

**注：从上面的描述可以看出，newton-cg, lbfgs 和 sag 这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的 L1 正则化，只能用于 L2 正则化。而 liblinear 通吃 L1 正则化和 L2 正则化。**

**liblinear 适用于小数据集，而 sag 和 saga 适用于大数据集因为速度更快。**

[XGBoost](https://xgboost.readthedocs.io/en/latest/)
====================================================

**XGBoost 的作者把所有的参数分成了三类：**

1.  通用参数：宏观函数控制。
2.  Booster 参数：控制每一步的 booster(tree/regression)。
3.  学习目标参数：控制训练目标的表现。  
    参见这篇博文[机器学习系列 (12)_XGBoost 参数调优完全指南（附 Python 代码）](https://blog.csdn.net/han_xiaoyang/article/details/52665396)  
    **要调节的参数有两种：树参数和 boosting 参数。learning rate 没有什么特别的调节方法，因为只要我们训练的树足够多 learning rate 总是小值来得好。**

**选择一个相对来说稍微高一点的 learning rate。一般默认的值是 0.1，不过针对不同的问题，0.05 到 0.2 之间都可以**  
**决定当前 learning rate 下最优的决定树数量**。它的值应该在 40-70 之间。记得选择一个你的电脑还能快速运行的值，因为之后这些树会用来做很多测试和调参。  
接着调节树参数来调整 learning rate 和树的数量。我们可以选择不同的参数来定义一个决定树，降低 learning rate，同时会增加相应的决定树数量使得模型更加稳健

1.  **固定 learning rate 和需要估测的决定树数量**
    
    为了决定 boosting 参数，我们得先设定一些**参数的初始值**，可以像下面这样：
    
    *   min_ samples_ split=500:  
        这个值应该在总样本数的 0.5-1% 之间，由于我们研究的是不均等分类问题，我们可以取这个区间里一个比较小的数，500。
        
    *   min_ samples_ leaf=50:  
        可以凭感觉选一个合适的数，只要不会造成过度拟合。同样因为不均等分类的原因，这里我们选择一个比较小的值。
        
    *   max_ depth=8: 根据观察数和自变量数，这个值应该在 5-8 之间。这里我们的数据有 87000 行，49 列，所以我们先选深度为 8。
        
    *   max_ features=’sqrt’: 经验上一般都选择平方根
        
    *   subsample=0.8: 开始的时候一般就用 0.8
        

**注意我们目前定的都是初始值，最终这些参数的值应该是多少还要靠调参决定。现在我们可以根据 learning rate 的默认值 0.1 来找到所需要的最佳的决定树数量，可以利用网格搜索（grid search）实现，以 10 个数递增，从 20 测到 80。**

2.  **树参数可以按照这些步骤调节：**
    1.  调节 max_depth 和 num_samples_split
    2.  调节 min_samples_leaf
    3.  调节 max_features

**需要注意一下调参顺序，对结果影响最大的参数应该优先调节，就像 max_depth 和 num_samples_split。**

3.  **调节子样本比例来降低 learning rate**
    
    接下来就可以调节子样本占总样本的比例  
    param_test5 = {‘subsample’:[0.6,0.7,0.75,0.8,0.85,0.9]}
    

[MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)  
参数说明:

1.  hidden_layer_sizes : 例如 hidden_layer_sizes=(100, 50)，表示有两层隐藏层，第一层隐藏层有 100 个神经元，第二层有 50 个神经元。
2.  activation : 激活函数,{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, 默认 relu
    *   identity：f(x) = x
    *   logistic：其实就是 sigmod,f(x) = 1 / (1 + exp(-x)).
    *   tanh：f(x) = tanh(x).
    *   relu：f(x) = max(0, x)
3.  solver： {‘lbfgs’, ‘sgd’, ‘adam’}, 默认 adam，用来优化权重

*   lbfgs：quasi-Newton 方法的优化器
*   sgd：随机梯度下降
*   adam： Kingma, Diederik, and Jimmy Ba 提出的机遇随机梯度的优化器  
    注意：默认 solver ‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs 收敛更快效果也更好。

4.  alpha :float, 可选的，默认 0.0001, 正则化项参数
5.  batch_size : int , 可选的，默认’auto’, 随机优化的 minibatches 的大小 batch_size=min(200,n_samples)，如果 solver 是’lbfgs’，分类器将不使用 minibatch
6.  learning_rate : 学习率, 用于权重更新, 只有当 solver 为’sgd’时使用，{‘constant’，’invscaling’, ‘adaptive’}, 默认 constant
    *   ‘constant’: 有’learning_rate_init’给定的恒定学习率
    *   ‘incscaling’：随着时间 t 使用’power_t’的逆标度指数不断降低学习率 learning_rate_ ，effective_learning_rate = learning_rate_init / pow(t, power_t)
    *   ‘adaptive’：只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损耗或验证分数停止升高至少 tol 时，将当前学习率除以 5.
7.  power_t: double, 可选, default 0.5，只有 solver=’sgd’时使用，是逆扩展学习率的指数. 当 learning_rate=’invscaling’，用来更新有效学习率。
8.  max_iter: int，可选，默认 200，最大迭代次数。
9.  random_state:int 或 RandomState，可选，默认 None，随机数生成器的状态或种子。
10.  shuffle: bool，可选，默认 True, 只有当 solver=’sgd’或者‘adam’时使用，判断是否在每次迭代时对样本进行清洗。
11.  tol：float, 可选，默认 1e-4，优化的容忍度
12.  learning_rate_int:double, 可选，默认 0.001，初始学习率，控制更新权重的补偿，只有当 solver=’sgd’ 或’adam’时使用。
13.  verbose : bool, 可选, 默认 False, 是否将过程打印到 stdout
14.  warm_start : bool, 可选, 默认 False, 当设置成 True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。
15.  momentum : float, 默认 0.9, 动量梯度下降更新，设置的范围应该 0.0-1.0. 只有 solver=’sgd’时使用.
16.  nesterovs_momentum : boolean, 默认 True, Whether to use Nesterov’s momentum. 只有 solver=’sgd’并且 momentum > 0 使用.
17.  early_stopping : bool, 默认 False, 只有 solver=’sgd’或者’adam’时有效, 判断当验证效果不再改善的时候是否终止训练，当为 True 时，自动选出 10% 的训练数据用于验证并在两步连续迭代改善，低于 tol 时终止训练。
18.  validation_fraction : float, 可选, 默认 0.1, 用作早期停止验证的预留训练数据集的比例，早 0-1 之间，只当 early_stopping=True 有用
19.  beta_1 : float, 可选, 默认 0.9，只有 solver=’adam’时使用，估计一阶矩向量的指数衰减速率，[0,1) 之间
20.  beta_2 : float, 可选, 默认 0.999, 只有 solver=’adam’时使用估计二阶矩向量的指数衰减速率 [0,1) 之间
21.  epsilon : float, 可选, 默认 1e-8, 只有 solver=’adam’时使用数值稳定值。

**属性说明：**

*   classes_: 每个输出的类标签
*   loss_: 损失函数计算出来的当前损失值
*   coefs_: 列表中的第 i 个元素表示 i 层的权重矩阵
*   intercepts_: 列表中第 i 个元素代表 i+1 层的偏差向量
*   n_iter_ ：迭代次数
*   n_layers_: 层数
*   n_outputs_: 输出的个数
*   out_activation_: 输出激活函数的名称。

参考文献：  
[https://xijunlee.github.io/2017/03/29/sklearn 中 SVM 调参说明及经验总结 /](https://xijunlee.github.io/2017/03/29/sklearn%E4%B8%ADSVM%E8%B0%83%E5%8F%82%E8%AF%B4%E6%98%8E%E5%8F%8A%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93/)  
[https://www.cnblogs.com/pinard/p/6065607.html](https://www.cnblogs.com/pinard/p/6065607.html)  
[https://blog.csdn.net/u011311291/article/details/78743393](https://blog.csdn.net/u011311291/article/details/78743393)