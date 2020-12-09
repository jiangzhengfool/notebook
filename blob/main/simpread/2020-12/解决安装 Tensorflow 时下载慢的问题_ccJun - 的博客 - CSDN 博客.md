> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/qq_38526623/article/details/105509341?utm_medium=distribute.pc_relevant.none-task-blog-baidulandingword-6&spm=1001.2101.3001.4242)

本文转载自：[https://blog.csdn.net/lixuminglxm/article/details/81386521](https://blog.csdn.net/lixuminglxm/article/details/81386521)

  在 TensorFlow 的[_中文版官方文档_](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/get_started/os_setup.md)中有详细的安装说明，具体步骤在此不再赘述。说一下自己在安装时的问题。  
  安装了 pip 工具之后，其默认的包下载路径为 python 官网，但下载速度龟慢，且连接不稳定，经常应为连接超时而失败。所以我们需要修改 pip 的下载源为国内的镜像库，常用的镜像库有阿里、豆瓣和清华等；  
  具体修改步骤为：  
  找到你系统下的 pip.conf 文件（若找不到，可以自己新建，我自己新建了一个，放在 / root/.pip / 下），并在其中添加如下内容：

```
[global]
index-url=http://pypi.douban.com/simple
extra-index-url=http://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host=
    pypi.douban.com
    mirrors.aliyun.com
```

  以上保存之后，则每次 pip 安装时都会先从 index-url 中查找，若没有则依次去其他 extra-index-url 中查找。

  若不想修改配置文件，则可以手动在命令行中指明要使用的镜像库：

```
pip install tensorflow -i https://pypi.douban.com/simple
```

若想安装指定版本：

> pip install tensorflow== 版本号 -i https://pypi.douban.com/simple