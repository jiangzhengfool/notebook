\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[blog.csdn.net\](https://blog.csdn.net/weixin\_42568012/article/details/105141281)

**目录**

[过采样](#%E8%BF%87%E9%87%87%E6%A0%B7)

[欠采样](#%E6%AC%A0%E9%87%87%E6%A0%B7)

* * *

样本分布不均衡将导致样本量少的分类所包含的特征过少，并很难从中提取规律；即使得到分类模型，也容易产生过度依赖于有限的数据样本而导致**过拟合**的问题，当模型应用到新的数据上时，模型的准确性和鲁棒性将很差。

抽样是解决样本分布不均衡相对简单且常用的方法，包括过采样和欠采样两种。过采样和欠采样**更适合大数据分布不均衡的情况**，尤其是第一种（过采样）方法应用更加广泛。

过采样
===

**增加分类中少数类样本的数量**来实现样本均衡，最直接的方法是简单复制少数类样本形成多条记录，这种方法的缺点是如果样本特征少而可能导致过拟合的问题；经过改进的过抽样方法通过在少数类中加入随机噪声、干扰数据或通过一定规则产生新的合成样本，例如 SMOTE 算法。

```
from imblearn.over\_sampling import RandomOverSampler  # 随机重复采样
from imblearn.over\_sampling import SMOTE  # 选取少数类样本插值采样
from imblearn.over\_sampling import BorderlineSMOTE  # 边界类样本采样
from imblearn.over\_sampling import ADASYN  # 自适应合成抽样
 
ros = RandomOverSampler(sampling\_strategy={0: 700,1:200,2:150 },random\_state=0)
X\_resampled, y\_resampled = ros.fit\_sample(X, y)
 
smo = SMOTE(sampling\_strategy={0: 700,1:200,2:150 },random\_state=42)
X\_smo, y\_smo = smo.fit\_sample(X, y)
 
bsmo = BorderlineSMOTE(kind='borderline-1',sampling\_strategy={0: 700,1:200,2:150 },random\_state=42) #kind='borderline-2'
X\_smo, y\_smo = bsmo.fit\_sample(X, y)
 
ana = ADASYN(sampling\_strategy={0: 800,2:300,1:400 },random\_state=0)
X\_ana, y\_ana = ana.fit\_sample(X, y)
```

欠采样
===

通过**减少分类中多数类样本的样本数量**来实现样本均衡，最直接的方法是随机地去掉一些多数类样本来减小多数类的规模，缺点是会丢失多数类样本中的一些重要信息。

```
from imblearn.under\_sampling import ClusterCentroids  # 根据K-means中心点生成
from imblearn.under\_sampling import RandomUnderSampler  # 随机选取数据子集
from imblearn.under\_sampling import NearMiss  # 根据近邻样本规则下采样 含3中方式
 
# 使用RandomUnderSampler方法进行欠抽样处理
model\_RandomUnderSampler = RandomUnderSampler() # 建立RandomUnderSampler模型对象
x\_RandomUnderSampler\_resampled, y\_RandomUnderSampler\_resampled = model\_RandomUnderSampler.fit\_sample(x,y) # 输入数据并作欠抽样处理
x\_RandomUnderSampler\_resampled = pd.DataFrame(x\_RandomUnderSampler\_resampled,columns=\['col1','col2','col3','col4','col5'\])
# 将数据转换为数据框并命名列名
y\_RandomUnderSampler\_resampled = pd.DataFrame(y\_RandomUnderSampler\_resampled,columns = \['label'\]) # 将数据转换为数据框并命名列名
RandomUnderSampler\_resampled = pd.concat(\[x\_RandomUnderSampler\_resampled, y\_RandomUnderSampler\_resampled\], axis= 1) # 按列合并数据框
groupby\_data\_RandomUnderSampler = RandomUnderSampler\_resampled.groupby('label').count() # 对label做分类汇总
print (groupby\_data\_RandomUnderSampler) # 打印输出经过RandomUnderSampler处理后的数据集样本分类分布
```