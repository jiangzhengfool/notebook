\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[blog.csdn.net\](https://blog.csdn.net/isoleo/article/details/46506275)

在 [linux](http://www.ttlsa.com/nginx/nginx-and-lua/ "Nginx 与 Lua") 系统管理中，我们有时候需要 kill 掉某个用户的所有进程，初学者一般先查询出用户的所有 pid，然后一条条 kill 掉，或者写好一个脚本，实际上方法都有现成的，这边有 4 种方法，我们以 kill 用户 ttlsa 为例.  
**1\. pkill 方式**

**2\. killall 方式**

**3\. ps 方式**  
ps 列出 ttlsa 的 pid，然后依次 kill 掉，比较繁琐.

**4\. pgrep 方式**  
pgrep -u 参数查出用户的所有 pid，然后依次 kill