> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/u014410989/article/details/90172737)

取消 jupyter notebook 的密码
=======================

*   1。终端输入：`jupyter notebook --generate-config` 会生成一个配置文件，成功后会显示文件路径（`/Users/kyousugi/.jupyter/jupyter_notebook_config.py`）
*   2。打开路径下的`jupyter_notebook_config.py`配置文件，找

到 jupyter_notebook_config.py 文件，打开，找到：

```
# c.NotebookApp.token = '<generated>'
```

取消掉注释，并且把其取值设为空：

```
c.NotebookApp.token = ' '
```

这样就再也不要输入密码才能使用 notebook 了

  
作者：sbill  
链接：https://www.jianshu.com/p/4ab87736d68a  
来源：简书  
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

[https://www.cnblogs.com/lianyingteng/p/7811126.html](https://www.cnblogs.com/lianyingteng/p/7811126.html)