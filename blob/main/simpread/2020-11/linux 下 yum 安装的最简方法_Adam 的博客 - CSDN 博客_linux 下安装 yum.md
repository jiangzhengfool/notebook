\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[blog.csdn.net\](https://blog.csdn.net/iamhuanggua/article/details/60140867)

Linux 下软件之间依赖关系很复杂，有时候安装一个软件需要花上好几天，99% 的时间都在安装对应版本的依赖。Yum 的出现大大简化了软件管理工作，可以让用户在无需关心软件依赖的情况下，方便的进行软件的下载、更新和使用。软件安装完成后，yum 还会帮助设置系统环境变量，大大简化了工作量。

Yum 的安装方式有很多种，网上的各种教程也是五花八门，编译安装需要依赖特定的 python 版本，有些会有依赖的版本冲突，折腾了好几个小时，找到了下面这个最简单可用的安装方式。

yum 最简方法安装
----------

*   下载最新的 yum-3.2.28.tar.gz 并解压

wget http://yum.baseurl.org/download/3.2/yum-3.2.28.tar.gz  
tar xvf yum-3.2.28.tar.gz

*   运行安装

touch /etc/ yum.conf

cd yum-3.2.28  
yummain.py install yum

*   更新系统，搞定收工

yum check-update  
yum update  
yum clean all

yum 基本用法
--------

*   查询系统装已经安装的软件信息

对于一个 rpm 包来说，都是有 "-" 和 "." 构成的，基本上有以下几部分组成： \* 包名 \* 版本信息 \* 发布版本号 \* 运行平台，当出现 noarch, 代表的是软件可以平台兼容

1）查询系统中已经安装的软件

rpm -qa

 2）查询一个已经安装的文件属于哪个软件包；

rpm -qf 文件名的绝对路径

3）查询已安装软件包都安装到何处；

软件名定义是: rpm 包去除平台信息和后缀后的信息

rpm -ql 软件名

4）查询一个已安装软件包的信息

rpm  -qi 软件名

5）查看一下已安装软件的配置文件；

rpm -qc 软件名

6）查看一个已经安装软件的文档安装位置：

rpm -qd 软件名

7）查看一下已安装软件所依赖的软件包及文件；

rpm -qR 软件名

*    对于未安装的软件包信息查询

1）查看一个软件包的用途、版本等信息；

rpm -qpi rpm 文件

2）查看一件软件包所包含的文件；

rpm -qpl rpm 文件

3）查看软件包的文档所在的位置；

rpm -qpd rpm 文件

4）查看一个软件包的配置文件；

rpm -qpc rpm 文件

5）查看一个软件包的依赖关系

rpm -qpR rpm 文件

*   软件包的安装、升级、删除等

1) 安装或者升级一个 rpm 包

rpm -ivh rpm 文件【安装】 rpm -Uvh rpm 文件【更新】

2) 删除一个 rpm 包

rpm -e 软件名

 如何需要不管依赖问题，强制删除软件，在如上命令其后加上 --nodeps

*   签名导入

rpm --import 签名文件 rpm --import RPM-GPG-KEY