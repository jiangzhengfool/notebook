\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[blog.csdn.net\](https://blog.csdn.net/blmoistawinde/article/details/80816179)

    TfidfVectorizer 可以把原始文本转化为 tf-idf 的特征矩阵，从而为后续的文本相似度计算，主题模型 (如 [LSI](https://blog.csdn.net/blmoistawinde/article/details/83446529))，文本搜索排序等一系列应用奠定基础。基本应用如：

```
#coding=utf-8
from sklearn.feature\_extraction.text import TfidfVectorizer
document = \["I have a pen.",
            "I have an apple."\]
tfidf\_model = TfidfVectorizer().fit(document)
sparse\_result = tfidf\_model.transform(document)     # 得到tf-idf矩阵，稀疏矩阵表示法
print(sparse\_result)
# (0, 3)	0.814802474667   # 第0个字符串，对应词典序号为3的词的TFIDF为0.8148
# (0, 2)	0.579738671538
# (1, 2)	0.449436416524
# (1, 1)	0.631667201738
# (1, 0)	0.631667201738
print(sparse\_result.todense())                     # 转化为更直观的一般矩阵
# \[\[ 0.          0.          0.57973867  0.81480247\]
#  \[ 0.6316672   0.6316672   0.44943642  0.        \]\]
print(tfidf\_model.vocabulary\_)                      # 词语与列的对应关系
# {'have': 2, 'pen': 3, 'an': 0, 'apple': 1}
```

但是要把它运用到中文上还需要一些特别的处理，故写此文分享我的经验。

**第一步：分词**

 中文不比英文，词语之间有着空格的自然分割，所以我们首先要进行分词处理，再把它转化为与上面的 document 类似的格式。这里采用著名的中文分词库 jieba 进行分词：

```
import jieba
text = """我是一条天狗呀！
我把月来吞了，
我把日来吞了，
我把一切的星球来吞了，
我把全宇宙来吞了。
我便是我了！"""
sentences = text.split()
sent\_words = \[list(jieba.cut(sent0)) for sent0 in sentences\]
document = \[" ".join(sent0) for sent0 in sent\_words\]
print(document)
# \['我 是 一条 天狗 呀 ！', '我 把 月 来 吞 了 ，', '我 把 日来 吞 了 ，', '我 把 一切 的 星球 来 吞 了 ，', '我 把 全宇宙 来 吞 了 。', '我 便是 我 了 ！'\]
```

PS：语料来自郭沫若《天狗》。另外，由于分词工具的不完善，也会有一些错误，比如这边错误地把 "日来" 分到了一起。

**第二步：建模**

 理论上，现在得到的 document 的格式已经可以直接拿来训练了。让我们跑一下模型试试。

```
tfidf\_model = TfidfVectorizer().fit(document)
print(tfidf\_model.vocabulary\_)
# {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}
sparse\_result = tfidf\_model.transform(document)
print(sparse\_result)
# (0, 4)	0.707106781187
# (0, 1)	0.707106781187
# (2, 5)	1.0
# (3, 6)	0.707106781187
# (3, 0)	0.707106781187
# (4, 3)	1.0
# (5, 2)	1.0
```

    没有错误，但有一个小问题，就是单字的词语，如 “我”、“吞”、“呀” 等词语在我们的词汇表中怎么都不见了呢？为了处理一些特殊的问题，让我们深入其中的一些参数。

**第三步：参数**

    查了一些资料以后，发现单字的问题是 **token\_pattern** 这个参数搞的鬼。它的默认值只匹配长度≥2 的单词，就像其实开头的例子中的'I'也被忽略了一样，一般来说，长度为 1 的单词在英文中一般是无足轻重的，但在中文里，就可能有一些很重要的单字词，所以修改如下：

```
tfidf\_model2 = TfidfVectorizer(token\_pattern=r"(?u)\\b\\w+\\b").fit(document)
print(tfidf\_model2.vocabulary\_)
# {'我': 8, '是': 12, '一条': 1, '天狗': 7, '呀': 6, '把': 9, '月': 13, '来': 14, '吞': 5, '了': 2, '日来': 10, '一切': 0, '的': 15, '星球': 11, '全宇宙': 4, '便是': 3}
```

 **token\_pattern** 这个参数使用正则表达式来分词，其默认参数为 r"(?u)\\b\\w\\w+\\b"，其中的两个 \\ w 决定了其匹配长度至少为 2 的单词，所以这边减到 1 个。对这个参数进行更多修改，可以满足其他要求，比如这里依然没有得到标点符号，在此不详解了。

    当然有些时候我们还是要过滤掉一些无意义的词，下面有些别的参数也可以帮助我们实现这一目的：

    1.**max\_df/min\_df: \[0.0, 1.0\] 内浮点数或正整数, 默认值 = 1.0**

   当设置为浮点数时，过滤出现在超过 max\_df / 低于 min\_df 比例的句子中的词语；正整数时, 则是超过 max\_df 句句子。

    这样就可以帮助我们过滤掉出现太多的无意义词语，如下面的 "我" 就被过滤（虽然这里 “我” 的排比在文学上是很重要的）。

```
\# 过滤出现在超过60%的句子中的词语
tfidf\_model3 = TfidfVectorizer(token\_pattern=r"(?u)\\b\\w+\\b", max\_df=0.6).fit(document)  
print(tfidf\_model3.vocabulary\_)
# {'是': 8, '一条': 1, '天狗': 5, '呀': 4, '月': 9, '来': 10, '日来': 6, '一切': 0, '的': 11, '星球': 7, '全宇宙': 3, '便是': 2}
```

    2.**stop\_words: list 类型**

      直接过滤指定的停用词。

```
tfidf\_model4 = TfidfVectorizer(token\_pattern=r"(?u)\\b\\w+\\b", max\_df=0.6, stop\_words=\["是", "的"\]).fit(document)
print(tfidf\_model4.vocabulary\_)
# {'一条': 1, '天狗': 5, '呀': 4, '月': 8, '来': 9, '日来': 6, '一切': 0, '星球': 7, '全宇宙': 3, '便是': 2}
```

    3.**vocabulary: dict 类型**

      只使用特定的词汇，其形式与上面看到的 tfidf\_model4.vocabulary\_相同，也是指定对应关系。

       这一参数的使用有时能帮助我们专注于一些词语，比如我对本诗中表达感情的一些特定词语（甚至标点符号）感兴趣，就可以设定这一参数，只考虑他们：

```
tfidf\_model5 = TfidfVectorizer(token\_pattern=r"(?u)\\b\\w+\\b",vocabulary={"我":0, "呀":1,"!":2}).fit(document)
print(tfidf\_model5.vocabulary\_)
# {'我': 0, '呀': 1, '!': 2}
print(tfidf\_model5.transform(document).todense())
# \[\[ 0.40572238  0.91399636  0.        \]
#  \[ 1.          0.          0.        \]
#  \[ 1.          0.          0.        \]
#  \[ 1.          0.          0.        \]
#  \[ 1.          0.          0.        \]
```

2019.12.21 再更新几个常用的参数 

 **4.ngram\_range: tuple**

      有时候我们觉得单个的词语作为特征还不足够，能够加入一些词组更好，就可以设置这个参数，如下面允许词表使用 1 个词语，或者 2 个词语的组合：

       这里顺便使用了一个方便的方法 get\_feature\_names() ，可以以列表的形式得到所有的词语

```
tfidf\_model5 = TfidfVectorizer(token\_pattern=r"(?u)\\b\\w+\\b", ngram\_range=(1,2), stop\_words=\["是", "的"\]).fit(document)
print(tfidf\_model5.get\_feature\_names())
"""
\['一切', '一切 星球', '一条', '一条 天狗', '了', '便是', '便是 我', '全宇宙', '全宇宙 来', '吞', '吞 了', '呀', '天狗', '天狗 呀', '我', '我 一条', '我 了', '我 便是', '我 把', '把', '把 一切', '把 全宇宙', '把 日来', '把 月', '日来', '日来 吞', '星球', '星球 来', '月', '月 来', '来', '来 吞'\]
"""
```

 **5.max\_feature: int**

       在大规模语料上训练 TFIDF 会得到非常多的词语，如果再使用了上一个设置加入了词组，那么我们词表的大小就会爆炸。出于时间和空间效率的考虑，可以限制最多使用多少个词语，模型会优先选取词频高的词语留下。下面限制最多使用 10 个词语：

```
tfidf\_model6 = TfidfVectorizer(token\_pattern=r"(?u)\\b\\w+\\b", max\_features=10, ngram\_range=(1,2), stop\_words=\["是", "的"\]).fit(document)
print(tfidf\_model6.vocabulary\_)
"""
{'我': 3, '把': 5, '来': 8, '吞': 1, '了': 0, '我 把': 4, '来 吞': 9, '吞 了': 2, '日来 吞': 6, '星球': 7}
"""
```

   比如这里大部分的词组都被过滤了，但是 “我 把” 因为多次出现而保留了。
