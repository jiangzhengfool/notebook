> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.heywhale.com](https://www.heywhale.com/api/notebooks/606fc41aae88cd00177a25ef/RenderedContent)

Task4 seaborn 可视化 (一)[¶](#Task4-seaborn可视化(一))
==============================================

*   Matplotlib 试着让简单的事情更加简单，困难的事情变得可能，而 Seaborn 就是让困难的东西更加简单。
*   seaborn 是针对统计绘图的，一般来说，seaborn 能满足数据分析 90% 的绘图需求。
*   Seaborn 其实是在 matplotlib 的基础上进行了更高级的 API 封装，从而使得作图更加容易，在大多数情况下使用 seaborn 就能做出很具有吸引力的图，应该把 Seaborn 视为 matplotlib 的补充，而不是替代物。
*   用 matplotlib 最大的困难是其默认的各种参数，而 Seaborn 则完全避免了这一问题。

*   本章知识点如下：
    *   导入相关库
    *   加载 seaborn 自带数据集：load_dataset()
    *   画布主题：set_style()
    *   关系类图表：relplot()
    *   散点图：scatterplot()
    *   折线图：lineplot()
    *   分类图表：catplot()
    *   分类散点图：stripplot() 和 swarmplot()
    *   箱图和增强箱图：boxplot() 和 boxenplot()
    *   小提琴图：violinplot()
    *   点图：pointplot()
    *   条形图：barplot()
    *   计数图：countplot()
*   小作业：
    *   第一题：绘制多个分类的散点图
    *   第二题：绘制 2010 年人口年龄结构金字塔
    *   第三题：绘制各年龄段男 VS 女占比差异线图

1. 导入相关库 [¶](#1.-导入相关库)
-----------------------

In [2]:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


```

2. 加载 seaborn 自带数据集 [¶](#2.-加载seaborn自带数据集)
-------------------------------------------

In [17]:

```
tips = pd.read_csv('/home/mw/input/seaborn6804/tips.csv')
fmri = pd.read_csv('/home/mw/input/seaborn6804/fmri.csv')

exercise = pd.read_csv('/home/mw/input/seaborn6804/exercise.csv')
titanic = pd.read_csv('/home/mw/input/seaborn6804/titanic.csv')


```

3. 画布主题 [¶](#3.-画布主题)
---------------------

*   关于如何调节图像的样式，这一块自由度太高就不具体介绍，就简单介绍一个修改背景的功能。画布主题共有五种类型：
    *   darkgrid：灰色网格
    *   whitegrid：白色网格
    *   dark：灰色
    *   white：白色
    *   ticks：这个主题比 white 主题多的是刻度线；
*   利用 set_style() 来修改，不过这个修改是全局性的，会影响后面所有的图像。

### 3.1 绘制三条 sin 函数曲线 [¶](#3.1-绘制三条sin函数曲线)

In [18]:

```
def sinplot(flip = 1):
    x = np.linspace(0, 10, 100)
    for i in range(1, 4):
        plt.plot(x, np.sin(x + i * 0.5) * (4 - i) * flip)
sinplot()


```

![](https://cdn.kesci.com/upload/rt/8FD1343D87F74E348C1B83A5EEC6A0DB/qraen09if2.png)

### 3.2 设置背景为默认主题 [¶](#3.2-设置背景为默认主题)

In [19]:

```
sns.set() # 默认主题：灰色网格；修改具有全局性
sinplot()


```

![](https://cdn.kesci.com/upload/rt/26645954DC4E4EDCB6FD02AFC2E84E84/qraen0u5x6.png)

### 3.3 修改背景为白色网格主题 [¶](#3.3-修改背景为白色网格主题)

In [20]:

```
sns.set_style('whitegrid')
sinplot()

#注：带有网格的主题便于读数


```

![](https://cdn.kesci.com/upload/rt/19D13A8063E944B28AF62690B2D57A7F/qraen184us.png)

### 3.4 去掉不必要的边框：sns.despine()[¶](#3.4-去掉不必要的边框：sns.despine())

In [21]:

```
sns.set_style("ticks")
sinplot() # 这个主题比white主题多的是刻度线
sns.despine() #去掉不必要的边框


```

![](https://cdn.kesci.com/upload/rt/2408EF117C3749D78AC277C2418CF49A/qraen1amjr.png)

*   去掉了上边框和右边框，despine() 还有别的参数，例如 offset 参数用于设置轴线偏移，更多参数可以自行搜索相关资料；

### 3.5 设置临时主题：内部白色网格，外部灰色主题 [¶](#3.5-设置临时主题：内部白色网格，外部灰色主题)

In [22]:

```
plt.figure(figsize = (10, 8))
sns.set_style('dark')
with sns.axes_style('whitegrid'): # with内部的都是白色网格主题，对外部不受影响
    plt.subplot(2, 1, 1) #绘制多图函数，两行一列第一个子图
    sinplot()
plt.subplot(2, 1, 2) # 两行一列第二个子图
sinplot()


```

![](https://cdn.kesci.com/upload/rt/D86083030EDD42AF86DB875450A3380F/qraen3fgsp.png)

### 3.6 标签与图形粗细调整：set_context()[¶](#3.6-标签与图形粗细调整：set_context())

*   当需要保存图表时，默认的参数保存下来的图表上刻度值或者标签有可能太小，有些模糊，可以通过 set_context() 方法设置参数。使保存的图表便于阅读。
*   有 4 种预设好的上下文 (context)，按相对大小排序分别是:
    *   paper
    *   notebook # 默认
    *   talk
    *   poster

In [23]:

```
sns.set()
plt.figure(figsize=(8,3))
sns.set_context("paper")
sinplot()

#其他参数自行尝试：


```

![](https://cdn.kesci.com/upload/rt/B9EC5EB952164B3484C1D5C73ED12981/qraen58zdr.png)

4. 关系类图表：relplot()[¶](#4.-关系类图表：relplot())
------------------------------------------

*   relplot() 关系类图表的接口，其实是下面两种图的集成，通过指定 kind 参数可以画出下面的两种图：
    *   scatterplot() 散点图
    *   lineplot() 折线图

### 4.1 基本的散点图 [¶](#4.1-基本的散点图)

In [24]:

```
sns.relplot(x="total_bill", y="tip", hue='day', data=tips)


```

Out[24]:

```
<seaborn.axisgrid.FacetGrid at 0x7fee867223c8>

```

![](https://cdn.kesci.com/upload/rt/DAFB7270582142A387187A815CCDEB57/qraenbx48c.png)

### 4.2 设置 col = 列的名称 则根据列的类别展示数据 (该列的值有多少种，则将图以多少列显示)[¶](#4.2-设置col=列的名称--则根据列的类别展示数据(该列的值有多少种，则将图以多少列显示))

In [25]:

```
sns.relplot(x="total_bill", y="tip", hue="day", col="time", data=tips)

#若设置row=列的名称 则根据列的类别展示数据(该列的值有多少种，则将图以多少行显示)


```

Out[25]:

```
<seaborn.axisgrid.FacetGrid at 0x7fee8661f240>

```

![](https://cdn.kesci.com/upload/rt/972BB739FE55434DB703ED020265235F/qraenioi8l.png)

### 4.3 布局：如果同时设置 col 和 row，则相同的 row 在同一行，相同的 col 在同一列 [¶](#4.3-布局：如果同时设置col和row，则相同的row在同一行，相同的col在同一列)

In [26]:

```
sns.relplot(x="total_bill", y="tip", hue="day", col="time", row="sex", data=tips)


```

Out[26]:

```
<seaborn.axisgrid.FacetGrid at 0x7fee8667ab38>

```

![](https://cdn.kesci.com/upload/rt/DF7090F0103544E482975F33464D48C0/qraenoqt5w.png)

5. 散点图：scatterplot()[¶](#5.-散点图：scatterplot())
----------------------------------------------

*   可以通过调整颜色、大小和样式等参数来显示数据之间的关系

### 5.1 绘制基本散点图 [¶](#5.1-绘制基本散点图)

In [27]:

```
sns.scatterplot(x="total_bill", y="tip", data=tips)


```

Out[27]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fee864fba20>

```

![](https://cdn.kesci.com/upload/rt/5F42D3356ABB4012B080C7DD47D2E99C/qraenuw0im.png)

### 5.2 设置 hue，根据设置的类别，产生颜色不同的点的散点图 [¶](#5.2-设置hue，根据设置的类别，产生颜色不同的点的散点图)

In [28]:

```
sns.scatterplot(x="total_bill", y="tip", hue="time", data=tips)


```

Out[28]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fee864da8d0>

```

![](https://cdn.kesci.com/upload/rt/4DE4EEA48F4245FD94271C0415BB165E/qraenxbxoe.png) In [29]:

```
tips.head()


```

Out[29]: .dataframe tbody tr th:only-of-type {vertical-align: middle;} .dataframe tbody tr th { vertical-align: top; } .dataframe thead th { text-align: right; }<table border="1"><thead><tr><th></th><th>total_bill</th><th>tip</th><th>sex</th><th>smoker</th><th>day</th><th>time</th><th>size</th></tr></thead><tbody><tr><th>0</th><td>16.99</td><td>1.01</td><td>Female</td><td>No</td><td>Sun</td><td>Dinner</td><td>2</td></tr><tr><th>1</th><td>10.34</td><td>1.66</td><td>Male</td><td>No</td><td>Sun</td><td>Dinner</td><td>3</td></tr><tr><th>2</th><td>21.01</td><td>3.50</td><td>Male</td><td>No</td><td>Sun</td><td>Dinner</td><td>3</td></tr><tr><th>3</th><td>23.68</td><td>3.31</td><td>Male</td><td>No</td><td>Sun</td><td>Dinner</td><td>2</td></tr><tr><th>4</th><td>24.59</td><td>3.61</td><td>Female</td><td>No</td><td>Sun</td><td>Dinner</td><td>4</td></tr></tbody></table>

### 5.3 设置 hue，根据设置的类别，产生颜色不同的点的散点图，设置 style，使其生成不同的标记的点 [¶](#5.3-设置hue，根据设置的类别，产生颜色不同的点的散点图，设置style，使其生成不同的标记的点)

In [30]:

```
sns.scatterplot(x="total_bill", y="tip", hue="day",, data=tips)


```

Out[30]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4dcca5f28>

```

![](https://cdn.kesci.com/upload/rt/12B22290B421483BB34DB5E7A8731EE1/qqvgx05hq1.png)

### 5.4 设置 size，根据设置的类别，产生大小不同的点的散点图 [¶](#5.4-设置size，根据设置的类别，产生大小不同的点的散点图)

In [31]:

```
sns.scatterplot(x="total_bill", y="tip", size="time", data=tips)


```

Out[31]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4c2b80550>

```

![](https://cdn.kesci.com/upload/rt/D022A8E5BBE44D3986E0F1C8050D6AF2/qqvgx0ucav.png)

### 5.5 使用指定的标记 [¶](#5.5-使用指定的标记)

In [32]:

```
markers = {"Lunch": "s", "Dinner": "X"}
sns.scatterplot(x="total_bill", y="tip",,
                markers=markers,
                data=tips)


```

Out[32]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4c2c2f390>

```

![](https://cdn.kesci.com/upload/rt/E96A49BD225C4514A41E957A74555B03/qqvgx1z1a1.png)

6. 折线图：lineplot()[¶](#6.-折线图：lineplot())
----------------------------------------

### 6.1 绘制带有误差带的单线图，显示置信区间 [¶](#6.1-绘制带有误差带的单线图，显示置信区间)

In [33]:

```
#seaborn自带的fmri数据集，加载数据集见第二部分；
sns.lineplot(x='timepoint', y='signal', data=fmri)


```

Out[33]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4c2c696d8>

```

![](https://cdn.kesci.com/upload/rt/2B06C2213C84492685684FBC25B5BC28/qqvgx1ddt3.png)

### 6.2 使用颜色和线型显示分组变量 [¶](#6.2-使用颜色和线型显示分组变量)

In [34]:

```
sns.lineplot(x='timepoint', y="signal",
             # 对将要生成不同颜色的线进行分组
             hue="region", 
             #对将生成具有不同破折号、或其他标记的变量进行分组
            , 
             #圆点标注
             marker='o',
             #数据集
             data=fmri)


```

Out[34]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa48435e5f8>

```

![](https://cdn.kesci.com/upload/rt/1E1F3DF8730546178E94B7AAABD645C7/qqvgx3bzcz.png)

### 6.3 显示错误条，而不显示错误带 [¶](#6.3-显示错误条，而不显示错误带)

In [35]:

```
sns.lineplot(x='timepoint', y="signal", hue="region", err_style="bars", ci=68, data=fmri)


```

Out[35]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4c2b4bac8>

```

![](https://cdn.kesci.com/upload/rt/DF581FB74BF54C918BCD164A627993CE/qqvgx4cmsz.png)

7. 分类图表：catplot()[¶](#7.-分类图表：catplot())
----------------------------------------

*   catplot() 分类图表的接口，其实是下面八种图表的集成，通过指定 kind 参数可以画出下面的八种图：
    *   stripplot() 分类散点图
    *   swarmplot() 能够显示分布密度的分类散点图
    *   boxplot() 箱图
    *   violinplot() 小提琴图
    *   boxenplot() 增强箱图
    *   pointplot() 点图
    *   barplot() 条形图
    *   countplot() 计数图

### 7.1 基本分类图 [¶](#7.1-基本分类图)

In [36]:

```
#seaborn自带exercise数据集,加载见第二部分；
sns.catplot(x="time", y="pulse", hue="kind", data=exercise)


```

Out[36]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa4c2bf2320>

```

![](https://cdn.kesci.com/upload/rt/560F2B2046474176885A91ECA88F8972/qqvgx44fax.png)

### 7.2 通过设置 kind 来指定绘制的图类型 [¶](#7.2-通过设置kind来指定绘制的图类型)

In [37]:

```
# kind="violin" 则表示绘制小提琴图
sns.catplot(x="time", y="pulse", hue="kind", data=exercise, kind="violin")


```

Out[37]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa484327cf8>

```

![](https://cdn.kesci.com/upload/rt/3D269EDB41D04BE6963F2AD8E019CD26/qqvgx4atg0.png)

### 7.3 根据 col 分类，以列布局绘制多列图 [¶](#7.3-根据col分类，以列布局绘制多列图)

*   设置 col，根据指定的 col 的变量名，以列的形式显示 (eg.col='diet', 则在列的方向上显示，显示图的数量为 diet 列中对值去重后的数量)

In [38]:

```
sns.catplot(x="time", y="pulse", hue="kind", col="diet", data=exercise)


```

Out[38]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa4e95526d8>

```

![](https://cdn.kesci.com/upload/rt/C26FCDC9912541108BD80C08FA6D1B74/qqvgx5uqzp.png)

### 7.4 绘图时，设置图的高度和宽度比 [¶](#7.4-绘图时，设置图的高度和宽度比)

In [39]:

```
sns.catplot(x="time", y="pulse", hue="kind",col="diet", data=exercise, height=4, aspect=.8)


```

Out[39]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa4c2b742e8>

```

![](https://cdn.kesci.com/upload/rt/46D0804BA42C467B8DCB96062B490CFB/qqvgx6wmim.png)

### 7.5 利用 catplot() 绘制柱状图 kind="count"[¶](#7.5-利用catplot()绘制柱状图-kind="count")

*   设置 col_wrap 一个数值，让图每行只显示数量为该数值的列，多余的另起一行显示

In [40]:

```
sns.catplot(x="time", col="kind", col_wrap=3, data=exercise, kind="count", height=2.5, aspect=.8)


```

Out[40]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa4841e3c50>

```

![](https://cdn.kesci.com/upload/rt/146ED34D410E48088DBCC27D65C28970/qqvgx6s246.png)

8. 分类散点图：stripplot() 和 swarmplot()[¶](#8.-分类散点图：stripplot()和swarmplot())
------------------------------------------------------------------------

### 8.1 分类散点图：stripplpt()[¶](#8.1-分类散点图：stripplpt())

*   stripplot() 可以自己实现对数据分类的展现，也可以作为盒形图或小提琴图的一种补充，用来显示所有结果以及基本分布情况。

In [41]:

```
sns.stripplot(x='day', y='total_bill', data=tips, jitter=True)
# jitter=False数据将会发生很多的重叠


```

Out[41]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481c0e390>

```

![](https://cdn.kesci.com/upload/rt/0FDF5ED47AE84CFFA74D83C8BA82D013/qqvgx7d3dm.png)

### 8.2 分簇散点图：swarmplot()[¶](#8.2-分簇散点图：swarmplot())

*   分簇散点图 可以理解为数据点不重叠的分类散点图。
*   该函数类似于 stripplot()，但该函数可以对点进行一些调整，使得数据点不重叠。
*   swarmplot() 可以自己实现对数据分类的展现，也可以作为盒形图或小提琴图的一种补充，用来显示所有结果以及基本分布情况。

In [42]:

```
sns.swarmplot(x='day', y='total_bill', data=tips)
# 数据不会发生重叠


```

Out[42]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481b7ca90>

```

![](https://cdn.kesci.com/upload/rt/116BBB8A5A1E41BCB1A3236F8999C6C7/qqvgx7l2ig.png) In [43]:

```
#hue对数据进行分类
sns.swarmplot(x='total_bill', y='day', data=tips, hue='sex')


```

Out[43]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481b5da90>

```

![](https://cdn.kesci.com/upload/rt/41014A5A30FF4709A6F6DCF9E8887110/qqvgx7djch.png)

9. 箱图和增强箱图：boxplot() 和 boxenplot()[¶](#9.-箱图和增强箱图：boxplot()和boxenplot())
------------------------------------------------------------------------

### 9.1 箱图：boxplot()[¶](#9.1-箱图：boxplot())

*   箱图又称盒形图，主要用来显示与类别相关的数据分布。

In [44]:

```
#使用seaborn自带的tips数据集，加载见第二部分；
plt.figure(figsize=(10,6))
sns.boxplot(x='day', y='total_bill', hue='time', data=tips)


```

Out[44]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481c8b9b0>

```

![](https://cdn.kesci.com/upload/rt/8C1D549F84024EDA845086C97155BE6A/qqvgx723qc.png)

### 9.2 箱型与分类散点组合图 [¶](#9.2-箱型与分类散点组合图)

In [45]:

```
plt.figure(figsize=(10,6))
sns.boxplot(x='day', y='total_bill', data=tips, palette='Purples_r')
sns.swarmplot(x='day', y='total_bill', data=tips)


```

Out[45]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481d65160>

```

![](https://cdn.kesci.com/upload/rt/43F982308F704B5F9CC86968D15B779F/qqvgx8h4ub.png)

### 9.3 利用 catplot() 来实现 boxplot() 的效果 (通过指定 kind="box")[¶](#9.3-利用catplot()来实现boxplot()的效果(通过指定kind="box"))

In [46]:

```
sns.catplot(x="sex", y="total_bill",
            hue="smoker", 
            col="time",
            data=tips, 
            kind="box",
            height=4, 
            aspect=.7);


```

![](https://cdn.kesci.com/upload/rt/AB6407FABBEB49A5A0729CC137682EE4/qqvgx8tbu6.png)

### 9.4 增强箱图：boxenplot()[¶](#9.4-增强箱图：boxenplot())

*   增强箱图又称增强盒形图，可以为大数据集绘制增强的箱图。
*   增强箱图通过绘制更多的分位数来提供数据分布的信息。

In [47]:

```
sns.boxenplot(x="day", y="total_bill", data=tips)


```

Out[47]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4817f06d8>

```

![](https://cdn.kesci.com/upload/rt/353ADEC347EF458A810AA59C12C16B10/qqvgx95cfu.png) In [48]:

```
#通过设置hue对分组数据进行第二次分类
#注意：在增强箱图中，对hue设置后的第二次分类的效果是分离
sns.boxenplot(x="day", y="total_bill", hue="smoker",
              data=tips, palette="Set3")


```

Out[48]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481865128>

```

![](https://cdn.kesci.com/upload/rt/B9A74AF45FD647EC9B33F72E57FCBF5F/qqvgx9lnkx.png)

### 9.5 利用 catplot() 来实现 boxenplot() 的效果 (通过指定 kind="boxen")[¶](#9.5-利用catplot()来实现boxenplot()的效果(通过指定kind="boxen"))

In [49]:

```
# 利用catplot()来实现boxenplot()的效果(通过指定kind="boxen")
sns.catplot(x="sex", y="total_bill",
            hue="smoker", 
            col="time",
            data=tips, 
            kind="boxen",
            height=4, 
            aspect=.7);


```

![](https://cdn.kesci.com/upload/rt/424BA5AE391040C782D7B5FE65B06187/qqvgxa4tar.png)

10. 小提琴图：violinplot()[¶](#10.-小提琴图：violinplot())
------------------------------------------------

*   小提琴图允许可视化一个或多个组的数字变量的分布。它与箱形图非常接近，但可以更深入地了解密度。小提琴图特别适用于数据量巨大且无法显示个别观察结果的情况。
*   小提琴图各位置对应参数，中间一条就是箱线图数据，25%，50%，75% 位置，细线区间为 95% 置信区间。

### 10.1 绘制按 sex 分类的小提琴图 [¶](#10.1-绘制按sex分类的小提琴图)

In [50]:

```
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips)
# 小提琴左右对称


```

Out[50]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481850c18>

```

![](https://cdn.kesci.com/upload/rt/F0617A8CFE69430E822203E98DD1A75F/qqvgxafimn.png)

### 10.2 添加 split 参数，使小提琴左右代表不同属性 [¶](#10.2-添加split参数，使小提琴左右代表不同属性)

In [51]:

```
sns.violinplot(x='day', y='total_bill', hue='smoker', data=tips, split=True)


```

Out[51]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481cceda0>

```

![](https://cdn.kesci.com/upload/rt/05F95DFDA89B4BBE8DF10AEEC9458558/qqvgxay0a2.png)

### 10.3 小提琴与分类散点组合图 [¶](#10.3-小提琴与分类散点组合图)

In [52]:

```
sns.violinplot(x='day', y='total_bill', data=tips, inner=None, palette='Set2')
sns.swarmplot(x='day', y='total_bill', data=tips, color='r', alpha=0.8)


```

Out[52]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481616710>

```

![](https://cdn.kesci.com/upload/rt/BAEE30C1366849AD877B531C32A13AB6/qqvgxb7o49.png)

### 10.4 使用 catplot() 实现 violinplot() 的统计效果，必须设置 kind="violin"[¶](#10.4-使用catplot()实现violinplot()的统计效果，必须设置kind="violin")

In [53]:

```
sns.catplot(x="day", y="total_bill",
               hue="sex",
               data=tips, 
               palette="Set2",
               split=True,
               scale="count", 
               inner="stick",
               scale_hue=False, 
               kind='violin',
               bw=.2)


```

Out[53]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa481616da0>

```

![](https://cdn.kesci.com/upload/rt/31CCB151E33F41DF9E7B4F8390286D7A/qqvgxbxdkp.png)

11. 点图：pointplot()[¶](#11.-点图：pointplot())
------------------------------------------

*   pointplot，如其名，就是点图。点图代表散点图位置的数值变量的中心趋势估计，并使用误差线提供关于该估计的不确定性的一些指示。
    
*   点图比条形图在聚焦一个或多个分类变量的不同级别之间的比较时更为有用。点图尤其善于表现交互作用：一个分类变量的层次之间的关系如何在第二个分类变量的层次之间变化。
    
*   重要的一点是点图仅显示平均值（或其他估计值），但在许多情况下，显示分类变量的每个级别的值的分布可能会带有更多信息。在这种情况下，其他绘图方法，例如箱型图或小提琴图可能更合适。
    

### 11.1 绘制点图，显示男女生存人数变化差异 [¶](#11.1-绘制点图，显示男女生存人数变化差异)

In [54]:

```
plt.figure(figsize=(8,4))
sns.pointplot(x='sex', y='survived', hue='class', data=titanic)


```

Out[54]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481616358>

```

![](https://cdn.kesci.com/upload/rt/E7DA7F5E4EEA44B78AEE790BCF0B582D/qqvgxc7098.png)

### 10.2 为点图增添些样式，使其更美观 [¶](#10.2-为点图增添些样式，使其更美观)

In [55]:

```
sns.pointplot(x='class', y='survived', hue='sex', data=titanic,
              palette={'male':'g', 'female': 'm'},        # 针对male和female自定义颜色
              markers=["^", "o"],     # 设置点的形状
              linestyles=["-", "--"]) # 设置线的类型


```

Out[55]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4811bee48>

```

![](https://cdn.kesci.com/upload/rt/6334FDF2A9F748F780A93E681750BE02/qqvgxctljj.png)

### 10.3 使用 catplot() 实现 pointplot() 的效果（通过设置 kind="point"）[¶](#10.3-使用catplot()实现pointplot()的效果（通过设置kind="point"）)

In [56]:

```
sns.catplot(x="sex", y="total_bill",
            hue="smoker", col="time",
            data=tips, kind="point",
            dodge=True,
            height=4, aspect=.7)


```

Out[56]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa4818b7fd0>

```

![](https://cdn.kesci.com/upload/rt/E69A21E5FA0F424996FEB1FF087E233B/qqvgxdgqh1.png)

11. 条形图：barplot()[¶](#11.-条形图：barplot())
----------------------------------------

*   条形图主要展现的是每个矩形高度的数值变量的中心趋势的估计。
*   条形图只显示平均值（或其他估计值）。但在很多情况下，每个分类变量级别上显示值的分布可能提供更多信息，此时很多其他方法，如一个盒子或小提琴图可能更合适。

### 11.1 指定 x 分类变量进行分组，y 为数据分布，绘制垂直条形图 [¶](#11.1-指定x分类变量进行分组，y为数据分布，绘制垂直条形图)

In [57]:

```
sns.barplot(x="day", y="total_bill", data=tips)


```

Out[57]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481196cf8>

```

![](https://cdn.kesci.com/upload/rt/F3D7B28740D0455F885E9A13F852C209/qqvgxdvza7.png)

### 11.2 指定 hue 对已分组的数据进行嵌套分组 (第二次分组) 并绘制条形图 [¶](#11.2-指定hue对已分组的数据进行嵌套分组(第二次分组)并绘制条形图)

In [58]:

```
sns.barplot(x="day", y="total_bill", hue="sex", data=tips)


```

Out[58]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa480feb278>

```

![](https://cdn.kesci.com/upload/rt/D33567481C124323BA232C2F229321DA/qqvgxe345s.png)

### 11.3 指定 y 为分类变量进行分组，x 为数据分布 (这样的效果相当于水平条形图)[¶](#11.3-指定-y-为分类变量进行分组，x-为数据分布-(这样的效果相当于水平条形图))

In [59]:

```
sns.barplot(x="tip", y="day", data=tips)


```

Out[59]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa4818fcdd8>

```

![](https://cdn.kesci.com/upload/rt/FF2ADE0799C54875893AB4459ED75181/qqvgxek53h.png)

### 11.4 设置 order=["变量名 1","变量名 2",...] 来显示指定分类顺序 [¶](#11.4-设置order=["变量名1","变量名2",...]来显示指定分类顺序)

In [60]:

```
sns.barplot(x="time", y="tip", data=tips,
            order=["Dinner", "Lunch"])


```

Out[60]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa481939588>

```

![](https://cdn.kesci.com/upload/rt/31AF36A6CBCB4F3D8A209A0E9E9345CD/qqvgxe3gt.png)

### 11.5 使用中位数作为集中趋势的估计：estimator=median[¶](#11.5-使用中位数作为集中趋势的估计：estimator=median)

In [61]:

```
sns.barplot(x="day", y="tip", data=tips, estimator=np.median)


```

Out[61]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa480fba4e0>

```

![](https://cdn.kesci.com/upload/rt/E1EB4865552A48078E6749924EE7C127/qqvgxefnol.png)

### 11.6 使用误差线显示均值的标准差 [¶](#11.6-使用误差线显示均值的标准差)

In [62]:

```
sns.barplot(x="day", y="tip", data=tips, ci=68)


```

Out[62]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa480fa1898>

```

![](https://cdn.kesci.com/upload/rt/E21676D447E244D9B466E391E821CEDE/qqvgxflrvy.png)

### 11.7 使用不同的调色版：palette="Blues_d"[¶](#11.7-使用不同的调色版：palette="Blues_d")

In [63]:

```
sns.barplot("size", y="total_bill", data=tips,
            palette="Blues_d")


```

Out[63]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa480fb1048>

```

![](https://cdn.kesci.com/upload/rt/F3FD2CD4DBC145EF826FF0DDC9869BF7/qqvgxf9tlo.png)

### 11.8 使用 catplot() 实现 barplot() 的效果 (通过指定 kind=bar)[¶](#11.8-使用catplot()实现barplot()的效果(通过指定kind=bar))

In [64]:

```
sns.catplot(x="sex", y="total_bill",
            hue="smoker", col="time",
            data=tips, kind="bar",
            height=4, aspect=.7)


```

Out[64]:

```
<seaborn.axisgrid.FacetGrid at 0x7fa4818f5fd0>

```

![](https://cdn.kesci.com/upload/rt/EDC60AE549ED4943846732289C24E94D/qqvgxf6gkt.png)

12. 计数图：countplot()[¶](#12.-计数图：countplot())
--------------------------------------------

*   seaborn.countplot() 可绘制计数图、柱状图
*   功能：使用条形图 (柱状图) 显示每个分类数据中的数量统计

### 12.1 显示单个分类变量的值统计数 [¶](#12.1-显示单个分类变量的值统计数)

In [65]:

```
sns.countplot(x="who", data=titanic)


```

Out[65]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa480e0f198>

```

![](https://cdn.kesci.com/upload/rt/EAB108900C2B41E98FBC73DF16ABF851/qqvgxftzai.png)

### 12.2 显示多个分类变量的值统计数 [¶](#12.2-显示多个分类变量的值统计数)

In [66]:

```
sns.countplot(x="class", hue="who", data=titanic)


```

Out[66]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa47cd9c4e0>

```

![](https://cdn.kesci.com/upload/rt/B6E4935506B24DD6B9B01BC79C102AFB/qqvgxfeeaz.png)

### 12.3 水平横向绘制条形图 [¶](#12.3-水平横向绘制条形图)

In [67]:

```
sns.countplot(y="class", hue="who", data=titanic)


```

Out[67]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa480e16a90>

```

![](https://cdn.kesci.com/upload/rt/46914E6434CD42CA8DE52C8AA08342C8/qqvgxgseja.png)

### 12.4 使用不同调色板 [¶](#12.4-使用不同调色板)

In [68]:

```
sns.countplot(x="who", data=titanic, palette="Set2")


```

Out[68]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fa47cd9c128>

```

![](https://cdn.kesci.com/upload/rt/4BA3D0D02B7C4079B542E77FEC029771/qqvgxgnkrv.png)

### 12.5 使用 catplot() 来实现 countplot() 的统计效果，必须设置 kind="count"[¶](#12.5-使用catplot()来实现countplot()的统计效果，必须设置kind="count")

In [69]:

```
sns.catplot(x="class", hue="who", col="survived",
            data=titanic, kind="count",
            height=4, aspect=.7);


```

![](https://cdn.kesci.com/upload/rt/1C4F53EC71BB4CBCAF0A4BE9737D9261/qqvgxgx25g.png)

小作业 [¶](#小作业)
-------------

### 第一题：绘制多个分类的散点图 [¶](#第一题：绘制多个分类的散点图)

*   要求：
    *   利用 pandas 构建时间序列数据，从 2000-1-31 开始，以月为频率，生成 100 条时间序列；
    *   生成 4 列 100 个服从高斯分布的随机数，并按列求累计和（cumsum 函数）；
    *   合并所有列，并设置列名为 a,b,c,d，生成散点图；

In [33]:

```
dt = pd.date_range(start="20000131", periods=100, freq="1M")
a = np.random.randn(100).cumsum()
b= np.random.randn(100).cumsum()
c = np.random.randn(100).cumsum()
d = np.random.randn(100).cumsum()
df = pd.DataFrame()
df['index'] = dt
df['a'] = a
df['b'] = b
df['c'] = c
df['d'] = d
# df.head(-1)
sns.scatterplot(x="index", y="a",  data=df)
sns.scatterplot(x="index", y="b",  data=df)
sns.scatterplot(x="index", y="c",  data=df)
sns.scatterplot(x="index", y="d",  data=df)


```

Out[33]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x7fee86430940>

```

![](https://cdn.kesci.com/upload/rt/0AC65DFB6B544A22829259A943473060/qraf1qoyaf.png) In [ ]:

### 第二题：绘制 2010 年人口年龄结构金字塔 [¶](#第二题：绘制2010年人口年龄结构金字塔)

*   步骤:
    *   读取文件：people.csv
    *   筛选数据：地区: 全国；性别：男和女；年龄段：不等于合计；
    *   添加一列：人口占比 = 每行的统计人数 / 总统计人数 * 100，并保留两位有效数字；
    *   绘制人口年龄结构金字塔（女左男右）：
        *   横轴：人口占比；纵轴：年龄段 (0 岁, 1-4 岁, 5-9 岁.....)
        *   tips: 金字塔实际相当于两个相反方向的水平条形图，然后以纵轴合并即可得到金字塔结构；
        *   注: 记住修改横轴刻度，两边对称显示；

In [ ]:

In [ ]:

### 第三题：绘制各年龄段男 VS 女占比差异线图 [¶](#第三题：绘制各年龄段男VS女占比差异线图)

*   步骤：
    *   构建所需数据框：各年龄段，男占比，女占比；
    *   占比差异即为：男占比 - 女占比；
    *   横轴：各年龄段；纵轴：占比差异；
    *   绘制线图
        *   添加横纵轴标签；
        *   添加标题；
        *   设置横轴标签旋转 45 度；

In [ ]: