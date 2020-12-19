> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/eagle89/article/details/77749196?utm_source=blogxgwz9&utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-6&spm=1001.2101.3001.4242)

[redis-benchmark 压力测试](http://www.cnblogs.com/jandison/p/7337442.html)
======================================================================

redis-benchmark 是 redis 官方提供的压测工具，安装好 redis 后，默认安装。使用简便。

语法：

Usage: redis-benchmark [-h <host>] [-p <port>] [-c <clients>] [-n <requests]> [-k <boolean>]

模拟 20 个客户端，100000 次请求

redis-benchmark -h 192.168.1.1 -p 6379 -n 100000 -c 20

模拟 1000000 次请求，生成 100000000 个 set 结构

redis-benchmark -t set -n 1000000 -r 100000000

模拟 ping，set，get 各 100000 次，结果输出到 csv 文件

redis-benchmark -t ping,set,get -n 100000 --csv

模拟 100000 次键 foo 的存储性能

redis-benchmark -n 100000 -q script load "redis.call('set','foo','bar')"

模拟一下十万次请求：  
redis-benchmark -n 100000 -q  
  
模拟一下百万级访问近百万 key：  
[root@vm-ArthurGuo-1 ~]# redis-benchmark -n 1000000 -r1000000 -q  
  
模拟一个万级用户的并发：  
 redis-benchmark -c 10000 -n 1000000 -r 1000000 -q  

redis-benchmark --help

Usage: redis-benchmark [-h <host>] [-p <port>] [-c <clients>] [-n <requests]> [-k <boolean>]  
 -h <hostname>      Server hostname (default 127.0.0.1)  -- 主机 ip 地址  -p <port>          Server port (default 6379) -- 端口  -s <socket>        Server socket (overrides host and port) --socket（如果测试在服务器上测可以用 socket 方式）  -c <clients>       Number of parallel connections (default 50) -- 客户端连接数  -n <requests>      Total number of requests (default 10000) -- 总请求数  -d <size>          Data size of SET/GET value in bytes (default 2) --set、get 的 value 大小  -dbnum <db>        SELECT the specified db number (default 0) -- 选择哪个数据库测试（一般 0-15）  -k <boolean>       1=keep alive 0=reconnect (default 1) -- 是否采用 keep alive 模式  -r <keyspacelen>   Use random keys for SET/GET/INCR, random values for SADD -- 随机产生键值时的随机数范围   Using this option the benchmark will expand the string __rand_int__   inside an argument with a 12 digits number in the specified range   from 0 to keyspacelen-1. The substitution changes every time a command   is executed. Default tests use this to hit random keys in the   specified range.  -P <numreq>        Pipeline <numreq> requests. Default 1 (no pipeline). --pipeline 的个数（如果使用 pipeline 会把多个命令封装在一起提高效率）  -q                 Quiet. Just show query/sec values -- 仅仅查看每秒的查询数  --csv              Output in CSV format -- 用 csv 方式输出  -l                 Loop. Run the tests forever -- 循环次数  -t <tests>         Only run the comma separated list of tests. The test -- 指定命令                     names are the same as the ones produced as output.  -I                 Idle mode. Just open N idle connections and wait. -- 仅打开 n 个空闲链接  
Examples:  
 Run the benchmark with the default configuration against 127.0.0.1:6379:    $ redis-benchmark  
 Use 20 parallel clients, for a total of 100k requests, against 192.168.1.1: **   $ redis-benchmark -h 192.168.1.1 -p 6379 -n 100000 -c 20  -- 测试 set、get、mset、sadd 等场景下的性能**  
 Fill 127.0.0.1:6379 with about 1 million keys only using the SET test: **   $ redis-benchmark -t set -n 1000000 -r 100000000  -- 测试 set 随机数的性能**  
 Benchmark 127.0.0.1:6379 for a few commands producing CSV output:    $ redis-benchmark -t ping,set,get -n 100000 --csv  -- 使用 csv 的输出方式测试  
 Benchmark a specific command line:    $ redis-benchmark -r 10000 -n 10000 eval 'return redis.call("ping")' 0 -- 测试基本命令的速度  
 Fill a list with 10000 random elements:    $ redis-benchmark -r 10000 -n 10000 lpush mylist __rand_int__  -- 测试 list 入队的速度  
 On user specified command lines __rand_int__ is replaced with a random integer  with a range of values selected by the -r option.  
**下面我就测下我的笔记本电脑的 redis 性能：** [root@db1 ~]# redis-benchmark -h 127.0.0.1 -p 6379 -n 100000 -c 20 ====== PING_INLINE ======   100000 requests completed in 1.09 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.86% <= 1 milliseconds 100.00% <= 2 milliseconds 100.00% <= 2 milliseconds 91659.03 requests per second  
====== PING_BULK ======   100000 requests completed in 1.07 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.94% <= 1 milliseconds 100.00% <= 1 milliseconds 93545.37 requests per second  
====== SET ======   100000 requests completed in 1.03 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.78% <= 1 milliseconds 100.00% <= 1 milliseconds 97087.38 requests per second  
====== GET ======   100000 requests completed in 1.10 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.81% <= 1 milliseconds 100.00% <= 1 milliseconds 90909.09 requests per second  
====== INCR ======   100000 requests completed in 1.09 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.86% <= 1 milliseconds 100.00% <= 1 milliseconds 91911.76 requests per second  
====== LPUSH ======   100000 requests completed in 1.07 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.85% <= 1 milliseconds 100.00% <= 1 milliseconds 93808.63 requests per second  
====== LPOP ======   100000 requests completed in 1.01 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.89% <= 1 milliseconds 100.00% <= 1 milliseconds 98522.17 requests per second  
====== SADD ======   100000 requests completed in 1.04 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.76% <= 1 milliseconds 100.00% <= 1 milliseconds 96153.85 requests per second  
====== SPOP ======   100000 requests completed in 1.11 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.92% <= 1 milliseconds 100.00% <= 1 milliseconds 90171.33 requests per second  
====== LPUSH (needed to benchmark LRANGE) ======   100000 requests completed in 1.09 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.82% <= 1 milliseconds 100.00% <= 1 milliseconds 92081.03 requests per second  
====== LRANGE_100 (first 100 elements) ======   100000 requests completed in 2.53 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.91% <= 1 milliseconds 100.00% <= 2 milliseconds 100.00% <= 2 milliseconds 39603.96 requests per second  
====== LRANGE_300 (first 300 elements) ======   100000 requests completed in 5.17 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
91.01% <= 1 milliseconds 99.94% <= 2 milliseconds 100.00% <= 2 milliseconds 19346.10 requests per second  
====== LRANGE_500 (first 450 elements) ======   100000 requests completed in 7.41 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
61.54% <= 1 milliseconds 98.36% <= 2 milliseconds 99.96% <= 3 milliseconds 100.00% <= 4 milliseconds 100.00% <= 4 milliseconds 13498.92 requests per second  
====== LRANGE_600 (first 600 elements) ======   100000 requests completed in 9.49 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
41.24% <= 1 milliseconds 91.89% <= 2 milliseconds 99.78% <= 3 milliseconds 100.00% <= 4 milliseconds 100.00% <= 4 milliseconds 10541.85 requests per second  
====== MSET (10 keys) ======   100000 requests completed in 1.68 seconds   20 parallel clients   3 bytes payload   keep alive: 1  
99.28% <= 1 milliseconds 100.00% <= 1 milliseconds 59382.42 requests per second 从以上可以看出，20 个客户端，每种场景均有 100000 次请求：ping、set、get、lpush、lpop、spop 等都达到 90000 多 rps，但 lrange 前 100、300、500 等就比较慢了，才 10000 多 rps。  
再测下 set 的速度： [root@db1 ~]# redis-benchmark -t set -n 1000000 -r 100000000 ====== SET ======   1000000 requests completed in 10.56 seconds   50 parallel clients   3 bytes payload   keep alive: 1  
98.65% <= 1 milliseconds 99.90% <= 2 milliseconds 99.99% <= 3 milliseconds 100.00% <= 3 milliseconds 94741.83 requests per second 每秒 94741 次，非常快  
再来测试下 list 的入队速度： [root@db1 ~]# redis-benchmark -r 100000 -n 100000 lpush mylist __rand_int__ ====== lpush mylist __rand_int__ ======   100000 requests completed in 0.97 seconds   50 parallel clients   3 bytes payload   keep alive: 1  
98.83% <= 1 milliseconds 100.00% <= 1 milliseconds 102774.92 requests per second 超过了 10w 次。