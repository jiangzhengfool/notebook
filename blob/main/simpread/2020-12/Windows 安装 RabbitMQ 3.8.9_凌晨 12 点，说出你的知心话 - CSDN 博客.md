> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/qq_25719689/article/details/109300733)

### 版本关系

1、RabbitMQ 依赖于 Erlang，需要先安装 Erlang  
2、Erlang 和 RabbitMQ 版本有对应关系  
http://www.rabbitmq.com/which-erlang.html

### 下载安装 Erlang 23.1

如果下载太慢了，可以把地址贴到迅雷里面，下载到本机

https://www.erlang.org/downloads/23.1  
exe 文件一路 next 就可以

### 配置 Erlang 环境变量

ERLANG_HOME=C:\Program Files\erl23.1  
Path 添加

%ERLANG_HOME%\bin;  
CMD 输入 erl，输入能显示版本号则安装正确

### 下载安装 RabbitMQ 3.8.9

http://www.rabbitmq.com/install-windows.html

### RabbitMQ 环境变量

RABBITMQ_SERVER=C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9  
在 Path 中加入

%RABBITMQ_SERVER%\sbin;  
启用 RabbitMQ 管理插件  
CMD 中输入

```
"C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmq-plugins.bat" enable rabbitmq_management
```

### 启动 RabbitMQ

net start RabbitMQ  
关闭 RabbitMQ

net stop RabbitMQ  
访问管理界面：http://localhost:15672/  
默认用户名：guest  
默认密码为：guest

默认配置文件：

```
C:\Users\你的用户名\AppData\Roaming\RabbitMQ\advanced.config
```

数据目录：

```
C:\Users\用户名\AppData\Roaming\RabbitMQ\db\rabbit@用户名-mnesia
```

附：  
如果遇到无法启动的问题，先尝试在控制面板 —— 服务 —— 中启动。  
如果已经启动了，先服务里面停掉  
或者尝试用命令

```
.\rabbitmq-server.bat start
```

注意只能用 CMD，不要用 powershell

如果要初始化 RabbitMQ，移除全部数据：

```
rabbitmq-service remove
rabbitmq-service install
rabbitmq-service start
```