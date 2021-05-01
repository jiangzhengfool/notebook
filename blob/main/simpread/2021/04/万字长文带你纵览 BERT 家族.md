> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/145119424)

以下文章来源于 NLP 有品，作者 NLPer 的老板娘

自 18 年底谷歌 BERT 问世以后，NLP 便逐渐步入 bert 时代，bert 家族儿孙满堂，如 RoBERTa、ALBert、ERNIE 等等，这些 bert 们正在给并持续给 nlp 领域输入无限生机，让人工智能皇冠上的明珠更加光彩夺目，在其光芒的照耀下，人类的人工智能之路必定越来越清晰、明朗。

通过阅读大量博客资料，知乎专栏和论文，文本以通俗易懂而不失专业的方式总结了 Bert 以及其 13 个衍生版本，分享给大家，不足之处，望请指出。后期会不定期分享各个版本 bert 的详细解读以及实战代码，敬请期待。

**1. BERT**
-----------

论文：《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》

论文地址：_[https://arxiv.org/pdf/1810.04805](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1810.04805)_

作者 / 机构：google

年份：2018.10

**1.1 概述**
----------

Bert 是基于 Transformer 的深度双向预训练语言模型，神经语言模型可直接用于下游 NLP 任务的微调。Bert 的出现使 finetune 技术应用于 NLP 领域真正走向成熟，并在工业界得到了广泛的应用，在不太消耗算力的情况下能显著提升各类任务的性能；在学术界也成为了研究热点，Bert 出现会后，基于 Bert 的各类任务的 SOTA 模型也逐渐出现，Bert 的改进版本也逐渐被提出。

**1.2 模型解读**
------------

模型由输入层（Embedding），编码层（Tansformer encoder）和输出层三部分组成。

![](https://pic4.zhimg.com/v2-322a4a8763ec7e63c64216dbf2e2a62f_r.jpg)

**（1）Embedding**

输入又三类特征相加而得，如下：

![](https://pic1.zhimg.com/v2-99c5277e107a893a9a19ce00ff172580_r.jpg)

**Token Embedding：**词特征（词向量）的嵌入，针对中文，目前只支持字特征嵌入；

**Segment Embedding：**词的句子级特征嵌入，针对双句子输入任务，做句子 A，B 嵌入，针对单句子任务，只做句子 A 嵌入；

**Position Embedding：**词的位置特征，针对中文，目前最大长度为 512；

**（2）Encoder**

使用的是具有强大特征提取能力的 Transformer 的编码器，其同时具有 RNN 提取长距离依赖关系的能力和 CNN 并行计算的能力。这两种能力主要是得益于 Transformer-encoder 中的 self-attention 结构，在计算当前词的时候同时利用了它上下文的词使其能提取词之间长距离依赖关系；由于每个词的计算都是独立不互相依赖，所以可以同时并行计算所有词的特征。Transformer 与 Transformer-encoder 结构如下：

![](https://pic3.zhimg.com/v2-7d62e63c89a6f7caf89dc154d481fe26_r.jpg)![](https://pic1.zhimg.com/v2-2f18fb341af007a13ac0dec5b38a4144_b.jpg)

**（3）输出层**

Bert 预训练的时候使用两大任务联合训练的，根据任务不同，其输出也不同，两大任务包括，掩码语言模型（MLM）任务和句子连贯性判定（NSP）任务其细节如下：

![](https://pic1.zhimg.com/v2-13af9cc641f15213d1ee32eef88018b8_b.jpg)

**MLM：**随机将输入中 15% 的词遮蔽起来，通过其他词预测被遮盖的词（这就是典型的语言模型），通过迭代训练，可以学习到词的上下文特征、语法结构特征、句法特征等，保证了特征提取的全面性，这对于任何一项 NLP 任务都是尤为重要。

其中，在选择 mask 的 15% 的词当中，80% 情况下使用 mask 掉这个词，10% 情况下采用一个任意词替换，剩余 10% 情况下保持原词汇不变。这么做的主要原因是：在后续微调任务中语句中并不会出现 [MASK] 标记，这样做可以削弱后续微调输入与预训练输入的不匹配为题；而且这么做的另一个好处是：预测一个词汇时，模型并不知道输入对应位置的词汇是否为正确的词汇（10% 概率），这就迫使模型更多地依赖于上下文信息去预测词汇，并且赋予了模型一定的纠错能力。

![](https://pic1.zhimg.com/v2-d29f43c38da29502feb5acb30437b138_b.jpg)

**NSP：**输入句子 A 和句子 B，判断句子 B 是否是句子 A 的下一句，通过迭代训练，可以学习到句子间的关系，这对于文本匹配类任务显得尤为重要。

**1.3 BERT 的特点**
----------------

**（1）真正的双向：**使用双向 Transformer，能同时利用当前单词的上下文信息来做特征提取，这与双向 RNN 分别单独利用当前词的上文信息和下文信息来做特征提取有本质不同，与 CNN 将上下文信息规定在一个限定大小的窗口内来做特征提取有本质不同；

**（2）动态表征：**利用单词的上下文信息来做特征提取，根据上下文信息的不同动态调整词向量，解决了 word2vec 一词多义的问题；

**（3）并行运算的能力：**Transformer 组件内部使用自注意力机制 (self- attention)，能同时并行提取输入序列中每个词的特征。

**（4）易于迁移学习：**使用预训练好的 BERT，只需加载预训练好的模型作为自己当前任务的词嵌入层，后续针对特定任务构建后续模型结构即可，不需对代码做大量修改或优化。

**1.4 用法**
----------

针对不同的 NLP 任务，有不同的使用方式，如下：

![](https://pic3.zhimg.com/v2-91a00a249a1b1175c58fb69d84035f1e_r.jpg)

**（a）句对分类**

判断两句子之间的关系，如句子语义相似度、句子连贯性判定等，其本质是文本分类。

输入：两句子；

输出：句子关系标签。

**（b）单句子文本分类**

判断句子属于哪个类别，如新闻自动分类、问题领域分类等。

输入：一个句子；

输出：输出句子类别标签。

**（c）抽取式问答**

给定问答和一段文本，从文本中抽取出问题的答案，如机器阅读理解等。其本质是序列标注。

输入：一个问题，一段文本；

输出：答案在文本中的索引。

**（d）单句子序列标注**

给输入句子的每个 token 打上目标标签，如分词、词性标注、实体识别等。

输入：一段文本；

输出：文本中每个 token 对应的标签。

针对 google 开源的中文 Bert 模型和源码，对两类任务做微调的用法如下：

**· 序列标注**

（1）加载预训练 Bert 模型；

（2）取输出字向量：embedding = bert_model.get_sequence_output()；

（3）然后构建后续网络。

**· 文本分类**

（1）加载预训练 BERT 模型；

（2）取输出句向量：output_layer=bert_model.get_pooled_output()；

（3）然后构建后续网络。

**2. BERT 的后代**
---------------

Bert 出现之后，研究者们开始不断对其进行探索研究，提出来各式的改进版，再各类任务上不断超越 Bert。针对 Bert 的改进，主要体现在增加训练语料、增添预训练任务、改进 mask 方式、调整模型结构、调整超参数、模型蒸馏等。下面对近年来 Bert 的改进版本的关键点做叙述。

**2.1 XL-Net**
--------------

论文：《XLNet: Generalized Autoregressive Pretraining for Language Understanding》

论文地址：_[https://arxiv.org/pdf/1906.08237v1](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1906.08237v1)_

作者 / 机构：CMU+google

年份：2019.6

![](https://pic2.zhimg.com/v2-8dd588dd1934f6be0458fab638cddd0d_r.jpg)

XL-NET 主要是通过改变 MLM 了训练的方式, 来提高 Bert 的性能，提出了自回归（AR，autoregressive）语言模型训练方法，另外还针对长文本任务将 transformer 替换为 transformer-xl 来提高微调长文本任务的性能。XL-NET 的两个改进点如下：

**（1）采用 AR 模型替代 AE 模型，解决 mask 带来的负面影响**

Bert 预训练过程中，MaskLM 使用的是 AE（autoencoding）方式，使用 mask 掉的词的上下文来预测该 mask 的词，而在微调阶段，输入文本是没有 MASK 的，这就导致预训练和微调数据的不统一，从而引入了一些人为误差。

而 XL-Net 使用的是 AR 方式，避免了采用 mask 标记位，且保留了序列的上下文信息，使用双流注意力机制实现的，巧妙的改进了 bert 与传统 AR 模型的缺点。

这样做的目的是：取消 mask 标记避免了微调时候输入与预训练输入不一致带来的误差问题。

**（2）引入 transformer-xl**

Bert 的编码单元使用的是普通的 Transformer，其缺点是输入序列的长度受最大长度限制，对于特别长的序列就会导致丢失一些信息。对于长序列 transfo rmer 的做法将长序列分为 N 个段，然后分别独立计算 N 个段，然后将 N 个段的结果拼接，并且每一次的计算都没法考虑到每一个段之间的关系。

而 transformer-xl 就能解决这个问题，其具体做法是：将长序列文本分为多个段序列，在计算完前一段序列后将得到的结果的隐藏层的值进行缓存，下个段序列计算的过程中，把缓存的值拼接起来再进行计算。

这样做的目的是：不但能保留长依赖关系还能加快训练，因为每一个前置片段都保留了下来，不需要再重新计算，在 transformer-xl 的论文中，经过试验其速度比 transformer 快了 1800 倍。

**2.2 RoBERTa**
---------------

论文：《RoBERTa：A Robustly Optimized BERT Pretraining Approach》

作者 / 机构：Facebook + 华盛顿大学

论文地址：_[https://arxiv.org/pdf/1907.11692](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1907.11692)_

年份：2019.7

RoBERTa 在训练方法上对 Bert 进行改进，主要体现在改变 mask 的方式、丢弃 NSP 任务、训练超参数优化和使用更大规模的训练数据四个方面。其改进点如下：

**（1）静态 Mask 变动态 Mask**

Bert 在整个预训练过程，选择进行 mask 的 15% 的 Tokens 是不变的，也就是说从一开始随机选择了这 15% 的 Tokens，之后的 N 个 epoch 里都不再改变了。这就叫做静态 Masking。

而 RoBERTa 一开始把预训练的数据复制 10 份，每一份都随机选择 15% 的 Tokens 进行 Masking，也就是说，同样的一句话有 10 种不同的 mask 方式。然后每份数据都训练 N/10 个 epoch。这就相当于在这 N 个 epoch 的训练中，每个序列的被 mask 的 tokens 是会变化的。这就叫做动态 Masking。

这样做的目的是：动态 mask 相当于间接的增加了训练数据，有助于提高模型性能。

**（2）移去 NSP 任务**

Bert 为了捕捉句子之间的关系，使用了 NSP 任务进行预训练，就是输入一对句子 A 和 B，判断这两个句子是否是连续的。两句子最大长度之和为 512。

RoBERTa 去除了 NSP，而是每次输入连续的多个句子，直到最大长度 512（可以跨文章）。这种训练方式叫做（FULL-SENTENCES），而原来的 Bert 每次只输入两个句子。

这样做的目的是：实验发现，消除 NSP 损失在下游任务的性能上能够与原始 BERT 持平或略有提高。这可能是由于 Bert 一单句子为单位输入，模型无法学习到词之间的远程依赖关系，而 RoBERTa 输入为连续的多个句子，模型更能俘获更长的依赖关系，这对长序列的下游任务比较友好。

**（3）更大的 mini-batch**

BERTbase 的 batch size 是 256，训练 1M 个 steps。RoBERTa 的 batch size 是 8k。

这样做的目的是：作者是借鉴了在了机器翻译中的训练策略，用更大的 batch size 配合更大学习率能提升模型优化速率和模型性能的现象，并且也用实验证明了确实 Bert 还能用更大的 batch size。

**（4）更多的训练数据，更长的训练时间**

借鉴 RoBERTa（160G）用了比 Bert（16G）多 10 倍的数据。性能确实再次彪升。当然，也需要配合更长时间的训练。

这样做的目的是：很明显更多的训练数据增加了数据的多样性（词汇量、句法结构、语法结构等等），当然能提高模型性能。

**2.3 ALBERT**
--------------

论文：《ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations》

论文地址：_[https://arxiv.org/pdf/1909.11942](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.11942)_

作者 / 机构：google

年份：2019.9

采用了全新的参数共享机制，反观其他升级版 BERT 模型，基本都是添加了更多的预训练任务，增大数据量等轻微的改动。这次 ALBERT 的改进，不仅提升了模型的整体效果再一次拿下来各项榜单的榜首，而且参数量相比 BERT 来说少了很多。

对于预训练模型来说，提升模型的大小是能对下游任务的效果有一定提升，然而如果进一步提升模型规模，势必会导致显存或者内存出现 OOM 的问题，长时间的训练也可能导致模型出现退化的情况。为了解决这些问题，Google 爸爸提出了 ALBERT，该模型提出了两种减少内存的方法，同时提升了训练速度，其次改进了 BERT 中的 NSP 的预训练任务。其改进点如下：

**（1）对 Embedding 进行因式分解**

在 BERT 中，词 embedding 维度（E）与 encoder 输出的维度（H）是一样的都是 768。但是 ALBERT 认为，词级别的 embedding 是没有上下文依赖的信息的，而隐藏层的输出值不仅包含了词本身的意思还包括一些上下文依赖信息，因此理论上来说隐藏层的表述包含的信息应该更多一些，所以应该让 H>>E 才比较合理。

另外在 NLP 任务中，通常词典都会很大，embedding matrix 的大小是 E×V，如果和 BERT 一样让 H=E，那么 embedding matrix 的参数量会很大，并且反向传播的过程中，更新的内容也比较稀疏，造成模型空间的浪费。

针对上述的两个点，ALBERT 采用了一种因式分解的方法来降低参数量，简单来说就是在输入与 Embedding 输出之间加了个线性映射。首先把 one-hot 向量映射到一个低维度的空间，大小为 E，然后再映射到一个高维度的空间，说白了就是先经过一个维度很低的 embedding matrix，然后再经过一个高维度 matrix 把维度变到隐藏层的空间内，从而把参数量从 O(V×H) 降低到了 O(V×E+E×H)，当 E<<H 时参数量减少的很明显。

这样做的目的是：拉开 E 和 H 的大小，减小模型大小的同时不降低模型整体性能。

**（2）跨层的参数共享**

BERT 模型用的结构单元是 Transformer，Transformer 中共享参数有多种方案，只共享全连接层，只共享 attention 层，ALBERT 结合了上述两种方案，全连接层与 attention 层都进行参数共享，也就是说共享 encoder 内的所有参数，同样量级下的 Transformer 采用该方案后实际效果略有下降，但是参数量减少了很多，训练速度也提升了很多。

这样做的目的是：减小模型参数，提升训练速度，且文中提到训练速度快之外，ALBERT 每一层的输出的 embedding 相比于 BERT 来说震荡幅度更小一些。可见参数共享其实是有稳定网络参数的作用的。

**（3）移去 NSP 任务，使用 SOP 任务**

BERT 的 NSP 任务实际上是一个二分类，训练数据的正样本是通过采样同一个文档中的两个连续的句子，而负样本是通过采用两个不同的文档的句子。该任务主要是希望能提高下游任务（如文本匹配）的效果，但是后续的研究发现该任务效果并不好。NSP 其实包含了两个子任务，主题预测（两个句子是否来源于同一类文章）与关系一致性（两个句子是否是承接关系）预测，但是主题预测相比于关系一致性预测简单太多了，并且在 MLM 任务中其实也有类型的效果。

因而，ALBERT 中，为了只保留一致性任务去除主题识别的影响，提出了一个新的任务—句子连贯性预测 sentence-order prediction（SOP），SOP 的正样本和 NSP 的获取方式是一样的，负样本把正样本的顺序反转即可。SOP 因为实在同一个文档中选的，其只关注句子的顺序并没有主题方面的影响。并且 SOP 能解决 NSP 的任务，但是 NSP 并不能解决 SOP 的任务，该任务的添加给最终的结果提升了一个点。

这样做的目的是：通过调整正负样本的获取方式去除主题识别的影响，使预训练更关注于句子关系一致性预测。

**（4）移除 dropout**

除了上面提到的三个主要优化点，ALBERT 的作者还发现一个很有意思的点，ALBERT 在训练了 100w 步之后，模型依旧没有过拟合，于是乎作者果断移除了 dropout，没想到对下游任务的效果竟然有一定的提升。这也是业界第一次发现 dropout 对大规模的预训练模型会造成负面影响。

**2.4 ELECTRA**
---------------

论文：《Efficiently Learning an Encoder that Classifies Token Replacements Accurately》

论文地址：_[https://openreview.net/attachment?id=r1xMH1BtvB&name=original_pdf](https://link.zhihu.com/?target=https%3A//openreview.net/attachment%3Fid%3Dr1xMH1BtvB%26name%3Doriginal_pdf)_

作者 / 机构：斯坦福 + google

年份：2019.11

ELECTRA 对 Bert 的改进最主要的体现在是提出了新的预训练任务和框架，把生成式的 Masked language model(MLM) 预训练任务改成了判别式的 Replaced token detection(RTD) 任务，判断当前 token 是否被语言模型替换过。模型总体结构如下：

![](https://pic1.zhimg.com/v2-cc4ae9033f546c4b985def548a74b3a4_r.jpg)

使用一个 MLM 的 Generator-BERT（生成器）来对输入句子进行更改，然后传给 Discriminator-BERT（判别器）去判断哪个词被改过。

**（1）训练方式**

生成器的训练目标还是 MLM（预测被 mask 的词是否是原词，目标空间大小是词表长度），判别器的训练目标是序列标注（判断每个 token 是真是假，目标空间大小是 2），两者同时训练，但判别器的梯度不会传给生成器，目标函数如下：

![](https://pic3.zhimg.com/v2-853327af7512f68173539f8076dc1bba_b.jpg)

其中，λ=50，因为判别器的任务相对来说容易些，loss 相对 MLM loss 会很小，因此加上一个系数λ，这也是多任务联合训练的惯用技巧。

**（2）训练策略**

a. 在优化判别器时计算了所有 token 上的 loss，而 BERT 的 MLM loss 时会忽略没被 mask 的 token。作者在后来的实验中也验证了在所有 token 上进行 loss 计算会提升效率和效果。

b. 作者设置了相同大小的生成器和判别器，在不共享权重下的效果是 83.6，只共享 token embedding 层的效果是 84.3，共享所有权重的效果是 84.4，最后选择只共享 Embedding 层参数。

c....

**2.5 ERNIE**
-------------

论文：《ERNIE: Enhanced Representation from kNowledge IntEgration》

论文地址：_[https://arxiv.org/pdf/1904.09223v1](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1904.09223v1)_

作者 / 机构：百度

年份：2019.3

![](https://pic4.zhimg.com/v2-d6f9bfeb1181fecf95d9d75b0b375b43_r.jpg)

ERNIE 对 Bert 的改进主要体现在 mask 的方式上，将中文单字的 mask 改为连续的实体词和短语 mask，事 Bert 能够学习到真实世界的语义知识信息，以此来提高 Bert 的性能。

另外，之后清华也出了一个版本的 ERNIE，它将知识图谱融入到语言模型的预训练之中，使用 TransE 来获取知识图谱中的实体向量，然后将实体向量嵌入到 BERT 中。其改进点如下：

**（1）mask 字改为 mask 词**

Bert 是随机 mask 输入序列中的字，这样能很简单地推测出字之间的搭配，这样会让本来应该有强相关的一些连在一起的字词，在训练时是割裂开来的。这对于中文文本中广泛包含多个字的实体、短语等单一的语义的词，俘获其语义信息是欠佳的。

因而 ERNIE 在输入为字的基础上，对输入序列中的短语和实体类的词实体词进行连续 mask，这样一来短语信息就会融入到字的 embedding 中了。

这样做的目的是：使模型能够学习到实体、短语的语义信息，训练完成后字的 embedding 就具有了实体、短语的语义信息了，这对于有大量短语、实体的文本任务（特别是实体识别任务）是非常友好。

**（2）使用很多知识类的中文语料进行预训练**

在 Bert 的基础上，ERNIE 预训练的语料引入了多源数据知识，包括了中文维基百科，百度百科，百度新闻和百度贴吧（可用于对话训练）。

这样做的目的是：使用多源数据，增大了数据的多样性，且多源数据中包含了海量事实类知识，预训练的模型能够更好地建模真实世界的语义关系。

**2.6 BERT-WWM**
----------------

论文：《Pre-Training with WholeWord Masking for Chinese BERT》

论文地址：_[https://arxiv.org/pdf/1906.08101](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1906.08101)_

作者 / 机构：讯飞 + 哈工大

年份：2019.7

BERT-WWM 对 Bert 的改进主要体现在 mask 的方式上，使用全词 mask。其改进点如下：

与百度 ERNIE 相比，BERT-WWM 不仅仅是连续 mask 实体词和短语，而是连续 mask 所有能组成中文词语的字。具体做法是，针对中文，如果一个完整的词的部分字被 mask，则同属该词的其他部分也会被 mask，即对组成同一个词的汉字全部进行 Mask，即为全词 Mask。

这样做的目的是：预训练过程中，模型能够学习到词的语义信息，训练完成后字的 embedding 就具有了词的语义信息了，这对各类中文 NLP 任务都是友好的。

**2.7 SpanBERT**
----------------

论文：《SpanBERT: Improving Pre-training by Representing and Predicting Spans》

论文地址：[https://arxiv.org/pdf/1907.10529](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1907.10529)

作者 / 机构：华盛顿大学 + 普林斯顿大学

年份：2019.8

![](https://pic4.zhimg.com/v2-91374cd0e9f4cb4ea9343c3ed2bb5ce3_r.jpg)

SpanBERT 对 Bert 的改进主要体现在对 mask 方式的改进，丢弃 NSP 任务和增加 SBO（Span Boundary Objective）任务。其改进点如下：

**（1）Span mask 方案**

Bert 是随机 mask 输入序列中的字，这样能很简单地推测出字之间的搭配，这样会让本来应该有强相关的一些连在一起的字词，在训练时是割裂开来的。难以建立词中各个字之间的关联信息。针对这一短板 Bert-wwm 与 ERNIE 分别对更改了 mask 策略，Bert-wwm 是 mask 所有能够连续组成词的字，ERNIE 是 mask 所有能够连续组成实体和短语的字。

而 SpanBERT 的做法是根据几何分布，先随机选择一段（span）的长度，之后再根据均匀分布随机选择这一段的起始位置，最后按照长度遮盖。文中使用几何分布取 p=0.2，最大长度只能是 10，利用此方案获得平均采样长度分布。

这样做的目的是：前不久的 MASS 模型，表明可能并不需要想 Bert-wmm 或者 ERNIE 那样 mask，随机遮盖可能效果也很好。

**（2）加入 SBO 训练目标**

Span Boundary Objective 是该论文加入的新训练目标，希望被遮盖 Span 边界的词向量，能学习到 Span 的内容。或许作者想通过这个目标，让模型在一些需要 Span 的下游任务取得更好表现。具体做法是，在训练时取 Span 前后边界的两个词，这两个词不在 Span 内，然后用这两个词向量加上 Span 中被遮盖掉词的位置向量，来预测原词。

这样做的目的是：增强了 BERT 的性能，为了让模型让模型在一些需要 Span 的下游任务取得更好表现，特别在一些与 Span 相关的任务，如抽取式问答。

**（3）去除 NSP 任务**

XLNet 中发现 NSP 不是必要的，而且两句拼接在一起使单句子不能俘获长距离的语义关系，所以作者剔除了 NSP 任务，直接一句长句做 MLM 任务和 SBO 任务。

这样做的目的是：剔除没有必要的预训练任务，并且使模型获取更长距离的语义依赖信息。

**2.8 TinyBERT**
----------------

论文：《TINYBERT:DISTILLINGBERTFORNATURALLAN-GUAGEUNDERSTANDING》

论文地址：_[https://arxiv.org/pdf/1909.10351](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.10351)_

作者 / 机构：华科 + 华为诺亚方舟实验室

年份：2019.9

![](https://pic3.zhimg.com/v2-05d700dacfbf2c63b07de53ddd6fb0a6_r.jpg)

TinyBert 通过对 Bert 编码器中的 Transformer 进行压缩，使用两段式学习框架在精度允许范围内节约了计算资源和推理速度。其改进点如下：

**（1）Transformer 蒸馏法**

为了在加快推理速度和降低模型大小的同时保持准确率，作者提出了一种新颖的 transformer 蒸馏法，这是为基于 transformer 的模型专门设计的知识蒸馏（knowledge distillation，KD）方法。

这样做的目的是：Bert 类的预训练语言模型通常计算开销大，内存占用也大，因此很难在一些资源紧张的设备上有效执行。通过这种新的 KD 方法，LargeBERT 模型中编码的大量知识可以很好地迁移到小型 TinyBERT 模型中，这对计算资源紧张的设备上运行是友好的。

**（2）两段式学习框架**

除了提出新的 transformer 蒸馏法之外，研究者还提出了一种专门用于 TinyBERT 的两段式学习框架，从而分别在预训练和针对特定任务的具体学习阶段执行 transformer 蒸馏。

这样做的目的是：TinyBERT 可以获取 LargeBERT 的通用和针对特定任务的知识。

**2.9 DistillBERT**
-------------------

论文：《DistilBERT, a distilled version of BERT: smaller,faster, cheaper and lighter》

论文地址：_[https://arxiv.org/pdf/1910.01108](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.01108)_

作者 / 机构：Hugging face

年份：2019.10

DistillBert 是在 Bert 的基础上用知识蒸馏技术训练出来的小型化 bert，通过 teacher model 的 soft target 作为 total loss 的一部分，以诱导 student model 的训练，实现 Bert 模型的知识迁移。其主要做了以下三点改进：

**（1）减小编码器层数**

使用 Bert-base 作为 teacher model，在 bert-base 的基础上将网络层数减半来构建 student model，具体做法是在 12 层 Transformer-encoder 的基础上每 2 层中去掉一层，最终将 12 层减少到了 6 层，且每一层都是用 teacher model 对应层的参数来做初始化的。

**（2）去掉了 token type embedding 和 pooler。（3）利用 teacher model 的 soft target 和 teacher model 的隐层参数来训练 student mdoel。**

这样做的目的是：在精度损失不大的情况下压缩模型大小提高其推理速度，更适应线上应用满足业务需求。

**2.10 sentence-BERT**
----------------------

论文：《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》

论文地址：_[https://arxiv.org/pdf/1908.10084](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1908.10084)_

作者 / 机构：达姆施塔特工业大学

年份：2019.8

![](https://pic3.zhimg.com/v2-3cba86537735cfa907ae6a9e318e925a_r.jpg)

Sentence-BERT 主要是解决 Bert 语义相似度检索的巨大时间开销和其句子表征不适用于非监督任务如聚类，句子相似度计算等而提出的。Sentence-BERT 使用鉴孪生网络结构，获取句子对的向量表示，然后进行相似度模型的预训练即为 sentence-BERT。其预训练过程主要包括如下步骤：

**（1）孪生网络获取句向量表示**

使用孪生网络结构，使用 Bert 进行 finetune 来进行句子的语义相似度模型的预训练，其具体做法是：将句子对输入到参数共享的两个 bert 模型中，将 Bert 输出句子的所有字向量进行平均池化（既是在句子长度这个维度上对所有字向量求均值）获取到每个句子的句向量表示。

**（2）分类器特征拼接**

然后将两向量的元素级的差值向量与这两个句向量进行拼接，最后接 softmax 分类器来训练句子对分类任务，训练完成后就得到了 sentence-Bert 语义相似度预训练模型。

这样做的目的是：减小 Bert 语义检索的巨大时间开销，并使其适用于句子相似度计算，文本聚类等非监督任务。

实验结果也正是如此，对于同样的 10000 个句子，我们想要找出最相似的句子对，只需要计算 10000 次，需要大约 5 秒就可计算完全。从 65 小时到 5 秒钟，检索速度天壤之别。

**2.11 K-BERT**
---------------

论文：《K-BERT: Enabling Language Representation with Knowledge Graph》

论文地址：_[https://arxiv.org/pdf/1909.07606v1](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.07606v1)_

作者 / 机构：北京大学 + 腾讯

年份：2019.9

![](https://pic2.zhimg.com/v2-0082d4a4ddc6ea4b66e97a1aa3ed7fc9_b.jpg)

K-BERT 主要是为了提升 BERT 在知识驱动任务上微调性能，由于通用语料训练的 BERT 模型在知识驱动型的任务上较大的领域差异，微调效果欠佳。K-BERT 通过将知识图谱的三元组信息引入到 BERT 的预训练中，使能够模型习得特殊领域的语义知识，以提升其在知识驱动型任务上的表现。K-BERT 对 BERT 的预训练过程做了如下步骤的改进：

**（1）句子树生成**

句子首先会经过一个知识层（Knowledge Layer）。其中知识层包括三大中文知识图谱：CN-DBpedia、知网（HowNet）和自建的医学知识图（MedicalKG），知识层会检索知识图谱，将与句子中词相关联的知识图谱中的三元组信息注入到句子中，形成一个富有背景知识的句子树（Sentence tree）。

**（2）句子树顺序信息的表达**

然后将句子树拉平, 更改 BERT 输入的位置嵌入方式，具体做法是：通过给每个 token 进行软位置 (Soft-position) 编码来表达句子树的顺序信息。

**（3）句子树信息编码**

为了将句子树中的结构信息引入到 BERT 中，其对 transformer-encoder 中的 self-attention 中进行改动，在计算 softmax 分数的时候在 QK 后多加了个可见矩阵 (Visible matrix)M，可见矩阵的加入使得对当前词进行编码时候，模型能‘看得见’当前词枝干上的信息, 而‘看不见’与当前词不相干的其他枝干上的信息, 以防不相干枝干在 self-attention 计算时候互相影响。

**2.12 SemBert**
----------------

论文：《Semantics-aware BERT for Language Understanding》

论文地址：_[https://arxiv.org/pdf/1909.02209](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.02209)_

作者 / 机构：上海交通大学 + 云从科技

年份：2019.9

![](https://pic1.zhimg.com/v2-9104da0d61fc9e37188ab03ce1cbca88_r.jpg)

SemBERT 是将语义角色标注（SRL，Semantic Role Labeling）信息结合进去，来提高 BERT 的性能。K-BERT 对 BERT 的预训练过程做了如下步骤的改进：

**（1）获取 SRL 标注**

使用目前最优的语义角色标注器 deep-srl 对句子进行语义信息标注。

**（2）多语义标签融合**

首先获取多种语义标签信息, 具体做法是: 对各个标签建立向量表，取向量，之后简单地用一个双向 GRU（BiGRU） 模型来获得深层的全局双向信息即可。

然后融合这些语义标签信息，具体做法是: 将上一步骤中深层 GRU 双向信息都拼接起来，然后接一个全连接层来实现多语义标签信息的融合。

**（3）对齐 SRL token 和 BERT token 的粒度**

由于 BERT 使用的是 BPE（Byte Pair Encoding）分词，会把词分成子词（subword）。于是就需要将子词向量映射成一个词向量。具体做法是: 在子词区域使用 CNN，然后进行 max pooling 来提取词向量。然后将 BERT outputs 与 srl 词向量进行拼接来做 BERT 预训练。

**2.13 StructBERT**
-------------------

论文：《STRUCTBERT: INCORPORATING LANGUAGE STRUCTURES INTO PRE-TRAINING FOR DEEP LANGUAGE UNDERSTANDING》

论文地址：_[https://arxiv.org/pdf/1908.04577](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1908.04577)_

作者 / 机构：阿里巴巴

年份：2019.9

StructBERT 是将语言结构信息融入进 Bert, 来提高其性能, 其主要是增加了两个基于语言结构的训练目标：词序 (word-level ordering) 重构任务和句序 (sentence-level ordering) 判定任务。

StructBERT 在于 Bert 现有的预训练任务 MLM 和 NSP 的基础上，新增了两个预训练任务：词序重建和句序判定任务, 分别如下:

![](https://pic3.zhimg.com/v2-262ace4c2894951ecfa834cad185aaa2_r.jpg)

**（1）词序重构**

从未被 mask 的序列中随机选择部分子序列（使用超参数 K 来确定子序列长度，论文选择的 K 值 = 3，即 trigram），将子序列中的词序打乱，让模型重建原来的词序。作者从重新排列的子序列中选择 5%，进行词序的打乱。

**（2）句序判定**

给定句子对 (S1, S2)，判断 S2 是否是 S1 的下一个句子，或上一个句子，或毫无关联的句子（从 NSP 的 0/1 分类变成了三分类问题）。采样时，对于一个句子 S，1/3 的概率采样 S 的下一句组成句对，1/3 的概率采样 S 的上一句组成句对，1/3 的概率随机采样一个其他文档的句子组成句对。

至此，Bert 的各个衍生版本就总结完毕了，还有一些针对特殊领域的 Bert 改进版如 finBERT、BERT-int 等没有总结，不是通用领域的所有没有总结在内，后期若有这方面的需求再总结也无伤大雅。