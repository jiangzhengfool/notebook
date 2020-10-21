1、查看目录下的文件列表：hadoop fs –ls [文件目录]
hadoop fs -ls -h /lance 

2、将本机文件夹存储至hadoop上：hadoop fs –put [本机目录] [hadoop目录] 
hadoop fs -put lance / 

3、在hadoop指定目录内创建新目录：hadoop fs –mkdir [目录] 
hadoop fs -mkdir /lance

4、在hadoop指定目录下新建一个文件，使用touchz命令：
hadoop fs -touchz /lance/tmp.txt 

5、将本机文件存储至hadoop上：hadoop fs –put [本机地址] [hadoop目录]
hadoop fs -put tmp.txt /lance #直接目录
hadoop fs -put tmp.txt hdfs://www.lance.com/lance #远程服务器地址

6、打开已存在文件：hadoop fs –cat [文件名称]
hadoop fs -cat /lance/tmp.txt 

7、重命名hadoop fs –mv [旧文件名] [新文件名]
hadoop fs -mv /tmp /tmp_bak #修改文件夹名

8、将hadoop上文件down至本机已有目录下：hadoop fs -get [文件目录] [本机目录]
hadoop fs -get /lance/tmp.txt /lance 

9、删除hadoop上文件：hadoop fs -rm [文件地址] 
hadoop fs -rm /lance/tmp.txt 

10、删除hadoop上指定文件夹（包含子目录等）：hadoop fs –rm -r [目录地址]
hadoop fs -rmr /lance 

11、将hadoop指定目录下所有内容保存为一个文件，同时下载至本机
hadoop dfs –getmerge /user /home/t

12、将正在运行的hadoop作业kill掉
hadoop job –kill  [jobId]
