> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [segmentfault.com](https://segmentfault.com/a/1190000015571891)

#### redis 自带的 redis-benchmark 工具

Redis 自带了一个叫 redis-benchmark 的工具来模拟 N 个客户端同时发出 M 个请求。 （类似于 Apache ab 程序）。你可以使用 redis-benchmark -h 来查看基准参数。

*   1 使用方法 redis-benchmark [-h <host>] [-p <port>] [-c <clients>] [-n <requests]> [-k <boolean>]

<table><thead><tr><th>序号</th><th>选项</th><th>描述</th><th>默认值</th></tr></thead><tbody><tr><td>1</td><td>-h</td><td>指定 redis server 主机名</td><td>localhost</td></tr><tr><td>2</td><td>-p</td><td>指定 redis 服务端口</td><td>6379</td></tr><tr><td>3</td><td>-s</td><td>指定服务器 socket</td><td></td></tr><tr><td>4</td><td>-c</td><td>指定并发连接数</td><td>50</td></tr><tr><td>5</td><td>-n</td><td>指定请求数</td><td>10000</td></tr><tr><td>6</td><td>-d</td><td>以字节形式指定 SET/GET 值的数值大小</td><td>2</td></tr><tr><td>7</td><td>-k</td><td>1=keepalive 0=reconnect</td><td>1</td></tr><tr><td>8</td><td>-r</td><td>SET/GET/INCR 使用随机 key, SADD 使用随机值</td><td></td></tr><tr><td>9</td><td>-P</td><td>通过管道传输 &lt;numreq&gt; 请求</td><td>1</td></tr><tr><td>10</td><td>-q</td><td>强制退出 redis. 仅显示 query/sec 值</td><td></td></tr><tr><td>11</td><td>-csv</td><td>以 csv 格式输出</td><td></td></tr><tr><td>12</td><td>-l</td><td>生成循环 永久执行测试</td><td></td></tr><tr><td>13</td><td>-t</td><td>仅运行以逗号分隔的测试命令列表</td><td></td></tr><tr><td>14</td><td>-I</td><td>Idle 模式, 仅打开 N 个 idle 连接并等待</td></tr></tbody></table>

```
[root@redis-test-slave ~ ]$ redis-benchmark --help
Usage: redis-benchmark [-h <host>] [-p <port>] [-c <clients>] [-n <requests]> [-k <boolean>]

 -h <hostname>      Server hostname (default 127.0.0.1)
 -p <port>          Server port (default 6379)
 -s <socket>        Server socket (overrides host and port)
 -a <password>      Password for Redis Auth
 -c <clients>       Number of parallel connections (default 50)
 -n <requests>      Total number of requests (default 100000)
 -d <size>          Data size of SET/GET value in bytes (default 2)
 --dbnum <db>        SELECT the specified db number (default 0)
 -k <boolean>       1=keep alive 0=reconnect (default 1)
 -r <keyspacelen>   Use random keys for SET/GET/INCR, random values for SADD
  Using this option the benchmark will expand the string __rand_int__
  inside an argument with a 12 digits number in the specified range
  from 0 to keyspacelen-1. The substitution changes every time a command
  is executed. Default tests use this to hit random keys in the
  specified range.
 -P <numreq>        Pipeline <numreq> requests. Default 1 (no pipeline).
 -e                 If server replies with errors, show them on stdout.
                    (no more than 1 error per second is displayed)
 -q                 Quiet. Just show query/sec values
 --csv              Output in CSV format
 -l                 Loop. Run the tests forever
 -t <tests>         Only run the comma separated list of tests. The test
                    names are the same as the ones produced as output.
 -I                 Idle mode. Just open N idle connections and wait.
```

```
Examples:

 Run the benchmark with the default configuration against 127.0.0.1:6379:
   # 运行默认配置下的测试
   $ redis-benchmark

 Use 20 parallel clients, for a total of 100k requests, against 192.168.1.1:
   # 指定并发数20,总请求数为10W,redis server主机IP为192.168.1.1
   $ redis-benchmark -h 192.168.1.1 -p 6379 -n 100000 -c 20

 Fill 127.0.0.1:6379 with about 1 million keys only using the SET test:
   # 测试SET随机数性能
   $ redis-benchmark -t set -n 1000000 -r 100000000

 Benchmark 127.0.0.1:6379 for a few commands producing CSV output:
   # 测试结果输出到csv
   $ redis-benchmark -t ping,set,get -n 100000 --csv

 Benchmark a specific command line:
   # 执行特定命令下的测试
   $ redis-benchmark -r 10000 -n 10000 eval 'return redis.call("ping")' 0

 Fill a list with 10000 random elements:
   # 测试list入队的速度
   $ redis-benchmark -r 10000 -n 10000 lpush mylist __rand_int__

 On user specified command lines __rand_int__ is replaced with a random integer
 with a range of values selected by the -r option.
```

*   2 实际测试过程
    
    *   redis-benchmark 默认参数下的测试

```
[root@redis-test-slave ~ ]$ redis-benchmark
        ====== PING_INLINE ======
          100000 requests completed in 0.83 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        120192.30 requests per second
        
        ====== PING_BULK ======
          100000 requests completed in 0.85 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        118203.30 requests per second
        
        ====== SET ======
          100000 requests completed in 0.80 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        125786.16 requests per second
        
        ====== GET ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        125944.58 requests per second
        
        ====== INCR ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        126903.55 requests per second
        
        ====== LPUSH ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        126262.62 requests per second
        
        ====== RPUSH ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        126103.41 requests per second
        
        ====== LPOP ======
          100000 requests completed in 0.80 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        99.97% <= 1 milliseconds
        100.00% <= 1 milliseconds
        125628.14 requests per second
        
        ====== RPOP ======
          100000 requests completed in 0.80 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        125786.16 requests per second
        
        ====== SADD ======
          100000 requests completed in 0.80 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        125786.16 requests per second
        
        ====== HSET ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        126103.41 requests per second
        
        ====== SPOP ======
          100000 requests completed in 0.80 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        125628.14 requests per second
        
        ====== LPUSH (needed to benchmark LRANGE) ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        126262.62 requests per second
        
        ====== LRANGE_100 (first 100 elements) ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        127388.53 requests per second
        
        ====== LRANGE_300 (first 300 elements) ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        127388.53 requests per second
        
        ====== LRANGE_500 (first 450 elements) ======
          100000 requests completed in 0.78 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        127551.02 requests per second
        
        ====== LRANGE_600 (first 600 elements) ======
          100000 requests completed in 0.79 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        126742.72 requests per second
        
        ====== MSET (10 keys) ======
          100000 requests completed in 0.77 seconds
          50 parallel clients
          3 bytes payload
          keep alive: 1
        
        100.00% <= 0 milliseconds
        129701.68 requests per second
```

#### 参考

```
http://www.redis.cn/topics/benchmarks.html
```