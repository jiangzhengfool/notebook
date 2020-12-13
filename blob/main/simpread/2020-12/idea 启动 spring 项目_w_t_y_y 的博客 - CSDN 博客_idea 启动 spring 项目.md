> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/w_t_y_y/article/details/100337840)

一、没有安装 tomcat：

1、点击![](https://img-blog.csdnimg.cn/20190902180005681.png)

进入：![](https://img-blog.csdnimg.cn/20190902180038313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

2、点 maven，点击左上角 + 号：

配置 spring 项目：

（1）在 parameters 的 command file 中输入：

```
org.mortbay.jetty:maven-jetty-plugin:6.1.26:run
```

（2）

![](https://img-blog.csdnimg.cn/20190902180200875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

在 runner 的 vm option 中输入

-Djetty.port=8000

即可。

3、关闭弹框，选择刚配置的项目，点击运行按钮即可：

![](https://img-blog.csdnimg.cn/20190902180308947.png)

二、安装了 tomcat：

1、点击 Edit  Configurations

![](https://img-blog.csdnimg.cn/20190902180005681.png)

  进入配置页面，再点左上角的 + 号新增 tomcat 项目：

![](https://img-blog.csdnimg.cn/20201125142636729.png)

  在弹出的选项中选择 Tomcat Server---->Local，如果没有 Tomcat Server 需要单独安装插件

    ![](https://img-blog.csdnimg.cn/20201125142756706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

  2、接下来是配置你的 tomcat 路径和项目信息：

   （1）配置 server 服务器：切换到第一个 tab，即 Server 下：

         1）配置 Application  server：就是配置 tomcat 服务器，点 configure 可以配置，如我的：

         ![](https://img-blog.csdnimg.cn/20201125143352980.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

       2）配置端口号和访问路径：注意这里 url 的端口号和下面的 port 需要保持一致

         ![](https://img-blog.csdnimg.cn/20201125143510270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

  （2）配置项目：切换到第二个 tab，即 Deployment 这个 tab 下，点 + 号，在弹出的项目中选择你的项目 war

          ![](https://img-blog.csdnimg.cn/20201125143815117.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

        如我选择的：

    ![](https://img-blog.csdnimg.cn/20201125143621716.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

   到这里配置就结束了，如果需要配置多个 tomcat 项目，重复上面的 1、2 步骤即可，可以选择同一个 tomcat，但是注意端口不能重复，如我使用一个 tomcat 新建了两个项目：

    ![](https://img-blog.csdnimg.cn/20201125143935477.png)

3、启动：在控制台下找到 services：

 ![](https://img-blog.csdnimg.cn/20201125144256880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dfdF95X3k=,size_16,color_FFFFFF,t_70)

可以看到这里有两个我已经配置好的项目，oss_service 和 activity，需要启动哪一个，就选中项目，点左边的启动或 debug 按钮即可，控制台的 server 下（如图）打印的是项目的日志，

几个 tomcat log 打印的是 tomcat 的日志。