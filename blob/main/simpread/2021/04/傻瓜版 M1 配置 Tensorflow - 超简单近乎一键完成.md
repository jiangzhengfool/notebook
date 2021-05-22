> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/367512283)

前文[《macOS M1(AppleSilicon) 安装 TensorFlow 环境》](https://zhuanlan.zhihu.com/p/349409718)介绍了如何给 M1 macOS 安装配置 Tensorflow 环境。但在那篇文章中，有太多的依赖包安装，而在安装的过程中，可能会因为顺序的问题，依赖的安装会相互替换掉，导致最终可能因为各种各样的问题而导致失败，于是本文将简化安装配置过程，使用本文方法，应该可以大大提高安装的成功率。

### 前期准备

安装 Tensorflow 前，需要有两个前置准备：

*   Miniforge conda 环境  
    
*   安装好 `Xcode Command Line Tools`  
    

Miniforge conda 的安装可以参考[《macOS M1(Apple Silicon) 安装配置 Conda 环境》](https://zhuanlan.zhihu.com/p/349295868)，这里就不再赘述了。

至于 `Xcode Command Line Tools` ，正常情况下在安装 Miniforge 的时候就已经安装好，如果没有，可以使用下面的命令进行安装：

```
xcode-select --install
```

接下来我们需要下载一个 [environment.yml](https://link.zhihu.com/?target=https%3A//raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml) 的文件，这个是 tensorflow 的依赖库列表文件。

以上步骤准备好之后，就可以开始安装。

### 安装过程

首先我们利用上面下载好的 environment.yml 文件来创建一个 conda 环境，命令具体如下：

```
conda env create --file=environment.yml --name=tf_mac
```

等待环境创建完成后，我们激活新创建的环境：

```
conda activate tf_mac
```

然后再使用 `pip` 安装好 Apple 提供的 `tensorflow` 库即可。(版本为 aplha3，截止于发稿时最新版本)

```
pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
```

安装完成后，可以进入 `ipython` 测试以下安装包是否已经成功安装。（import tensorflow 的时候，可能有点慢，一般等待以下即可）

```
ipython
In [1]: import tensorflow

In [2]: tensorflow.__version__
Out[2]: '2.4.0-rc0'
```

这样就完成了 M1 的 tensorflow 配置了。如果手上没有现成的 tensorflow 项目，可以使用文末提供的代码跑一下测试。

### 小结

使用本文的方法，几乎是一键即可完成安装配置，耗时和步骤都相对于之前有很大的优化。希望本文对你有用。如果你觉得文章对你用，记得关注收藏。你的关注和收藏是继续更新的动力哦。

### 【附】测试代码

下载代码，运行即可，运行前，请先安装依赖：

```
pip install tensorflow-datasets==2.1.0

python cnn_benchmark.py
```

代码：

```
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import time
from datetime import timedelta

tf.enable_v2_behavior()

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

batch_size = 128

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                 activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#   tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

start = time.time()

model.fit(
    ds_train,
    epochs=10,
    # validation_steps=1,
    # steps_per_epoch=469,
    # validation_data=ds_test
)

delta = (time.time() - start)
elapsed = str(timedelta(seconds=delta))
print('Elapsed Time: {}'.format(elapsed))
```