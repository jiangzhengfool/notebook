> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/u010317829/article/details/52119281)

Zookeeper 在 Windows 下的安装过程及测试
=============================

1.  安装 jdk
2.  安装 Zookeeper. 在官网 [http://zookeeper.apache.org/](http://zookeeper.apache.org/) 下载 zookeeper. 我下载的是 zookeeper-3.4.6 版本。
3.  解压 zookeeper-3.4.6 至 D:\machine\zookeeper-3.4.6.
4.  在 D:\machine 新建 data 及 log 目录。
5.  ZooKeeper 的安装模式分为三种，分别为：单机模式（stand-alone）、集群模式和集群伪分布模式。ZooKeeper 单机模式的安装相对比较简单，如果第一次接触 ZooKeeper 的话，建议安装 ZooKeeper 单机模式或者集群伪分布模式。
    
6.  安装单击模式。 至 D:\machine\zookeeper-3.4.6\conf 复制 zoo_sample.cfg 并粘贴到当前目录下，命名 zoo.cfg.
    
7.  编辑 zoo.cfg. 修改如下配置  
    ![](https://img-blog.csdn.net/20160804144808278)
    
8.  cmd 命令下进入 D:\machine\zookeeper-3.4.6\bin 目录下运行 zkserver.cmd. 如下图所示：  
    ![](https://img-blog.csdn.net/20160804145122967)
    
9.  启动完成后 cmd 命令下，netstat-ano 查看端口监听服务。
10.  cmd 下进入 D:\machine\zookeeper-3.4.6\bin 目录下运行 zkcli.cmd. 如下图所示：  
    ![](https://img-blog.csdn.net/20160804145348921)
11.  安装集群伪分布模式。
12.  修改 zoo.cfg 文件。如下图所示：  
    ![](https://img-blog.csdn.net/20160804150352378)  
    另存为 zoo-1.cmd.  
    ![](https://img-blog.csdn.net/20160804150616660)  
    另存为 zoo-2.cmd.  
    ![](https://img-blog.csdn.net/20160804150710145)  
    另存为 zoo-3.cmd.
    
13.  修改 zkserver.cmd 文件。如下图所示：  
    ![](https://img-blog.csdn.net/20160804151042037)  
    另存为 zkserver-1.cmd  
    ![](https://img-blog.csdn.net/20160804151203148)  
    另存为 zkserver-2.cmd  
    ![](https://img-blog.csdn.net/20160804151219742)  
    另存为 zkserver-3.cmd。
    
14.  cmd 下分别运行 zkserver-1.cmd,zkserver-2.cmd,zkserver-3.cmd.
    
15.  cmd 下 netstar-ano 查看端口监听情况。
16.  cmd 下运行 zkcli.cmd -server:localhost:2181;zkcli.cmd ;-server:localhost:2182;zkcli.cmd -server:localhost:2183.
    
17.  zookeeper 与 java 的连接  
    ![](https://img-blog.csdn.net/20160804153622041)  
    单机连接：  
    ![](https://img-blog.csdn.net/20160804153641041)  
    集群连接：  
    ![](https://img-blog.csdn.net/20160804153650674)