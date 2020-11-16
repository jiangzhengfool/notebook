\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[www.jianshu.com\](https://www.jianshu.com/p/51dcd88168c5)

[![](https://upload.jianshu.io/users/upload_avatars/1957277/1a1cb45a-8012-4ab9-ad3f-3ad04d6b29a3.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/96/h/96/format/webp)](https://www.jianshu.com/u/12648978d547)

0.5562020.08.16 11:22:37 字数 1,155 阅读 201

学了一段时间的 Python 了，上周尝试写了个爬虫练手。在这里分享一下。这次要爬取的是一个图片网站。很多萌妹子图片哦！偷偷背着女朋友爬的，怕被打。纯属学习，不做商业用途。网址：[https://www.vmgirls.com/](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.vmgirls.com%2F)  
谷歌浏览器打开网页。

*   ![](http://upload-images.jianshu.io/upload_images/1957277-3cc81de2befb05e3.png)
    
    网站首页. png
    

网页滚到分组图片，今天我们就爬取最新分组图片

![](http://upload-images.jianshu.io/upload_images/1957277-852552c64ac1bf10.png)

网站首页最新组图片. png

然后按 F12 打开网页检查（或者鼠标右键单击点开检查），打开后如下图款中所示

*   ![](http://upload-images.jianshu.io/upload_images/1957277-5de18684c5722b96.png)
    
    image.png
    

然后刷新网页。  
网页加载后，我们查看框中的信息。点击类型是 document 的请求。

![](http://upload-images.jianshu.io/upload_images/1957277-a119813fd1db6ff6.png)

image.png

点击后发现了，请求方法请求头等相关信息。

![](http://upload-images.jianshu.io/upload_images/1957277-57c6ca035a1917bd.png)

image.png

继续往下滚动，发现了 “user-agent” 信息，这个可以用来模拟人工网页浏览的行为。

![](http://upload-images.jianshu.io/upload_images/1957277-2662d938c1f5ee62.png)

image.png

到目前为止已经具备了基本的爬取网页内容条件了。  
下面我们就用 "requests" 库进行爬取网页内容。  
打开 pycharm，coding 开始。导入 “requests” 库。顺便把等下要用的库一起导入进来。如下

```
import requests  # 导入requests库,主要做网络请求
import re  # 导入正则表达式库，用于正则提取目标内容
import os  # 导入操作系统库
import time  # 导入时间库
```

我们先定义一个类就叫 “WeiMeiGirls” 类  
然后定义一个初始化方法，初始化方法里面，我们定义好网址，请求头信息，主要是 “User-Agent”。一个“run” 方法

```
class WeiMeiGirls():
    def \_\_init\_\_(self):
        self.url\_main = "https://www.vmgirls.com/"
        self.user\_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'
            }

    def run(self):
           pass//run方法里面内容暂时不实现，先用pass，等下再完善
```

接下来我们接着写网页请求方法获取网页内容。

```
def get\_html\_main(self):
        response\_main = requests.get(self.url\_main, headers=self.user\_headers)
        assert response\_main.status\_code == 200  # 当响应码不是200时候，做断言报错处理
        html\_main = response\_main.text
        return html\_main
```

这个方法可以返回网页内容。通过打印网页信息，我们发现返回的是 HTML 内容。部分 HTML 内容截图如下。

![](http://upload-images.jianshu.io/upload_images/1957277-d99e5a21d218a9ea.png)

image.png

有了这些内容之后，我们就可以通过正则表达式获取到我们目标内容了。  
接下来我们通过谷歌浏览器查看这些图片组的规律。谷歌浏览器左上角有个箭头工具，可以用来产值渲染后网页的原始 HTML 元素。如下图所示。

![](http://upload-images.jianshu.io/upload_images/1957277-4fa2cfe611658148.png)

image.png

通过观察我们发现了这些组图原来在一个 class 为 "row row-sm list-home list-grouped list-bordered-padding my-n2" 大的 <div> 标签中。大的 < div > 标签中有嵌套了一组组的小的 < div>

![](http://upload-images.jianshu.io/upload_images/1957277-0f6a85d268129f27.png)

image.png

再进行分析这些小的 div 标签。

![](http://upload-images.jianshu.io/upload_images/1957277-23e72252a074c8c6.png)

image.png

我们发现原来这些 <div> 标签中有，我们想要的内容，组图标题，组图封面图地址。还有一个很重要的就是组图详情页，详情页就是组图里所有的图片。所以现在我们知道了，只要获取这三个内容就可以。经过观察发现详情页中也包含组图封面，还有组图标题，所以只要获取详情页网址就可以了。  
下面就开始写正则匹配需要的的数据。这个网址是写在一个 <a> 标签中，所以正则可以如下这样写。下面这个方法是获取组图详情页网址的。注意，每次的网络请求最好延迟 10 秒钟。不要请求过快。不要把服务器弄卡，占用过高的资源。

```
\# 对请求返回内容进行正则表达式 获取组图详情页网址，列表返回。
    def get\_detail\_url(self,html):
        url\_detail\_content = re.findall('<a class="media-content" href="(.\*?)" title="(.\*?) data-bg="(.\*?)">', html)
        url\_detail\_list=\[\]
        for itme in url\_detail\_content:
            url\_detail = itme\[0\]
            url\_detail\_list.append(url\_detail)
        return url\_detail\_list
```

有了详情页网址列表，我们再定义一个方法来获取下载图片

```
\# 获取图片组名称，图片下载链接，下载并保存图片
    def get\_pictures\_groupname(self,url\_details):
# for循环 到链接列表获取到 链接，然后进行request请求
        for detail\_url in url\_details:
            # 拼接详情页网址
            detail\_url = self.url\_main + detail\_url
            # 请求详情页
            respons\_detail = requests.get(detail\_url, headers=self.user\_headers)
            assert respons\_detail.status\_code == 200
            html\_detail = respons\_detail.text

            # 对请求返回内容进行正则表达式 获取图片下载链接
            url\_detail\_list = re.findall('<a href="(.\*?)" alt=".\*?" title=".\*?">', html\_detail)
            # print(url\_detail\_list)
            # 获取图片组名称
            groupname = re.findall('<a href=".\*?" alt=".\*?" title="(.\*?)">', html\_detail)
            print(groupname\[0\])
            time.sleep(5)  # 设定5秒延时
```

通过上面已经或到了图片组名称和图片下载链接分别放到了两个列表中，一个是图片链接列表，一个是图片组名称列表，图片组名称取下标为 0 的元素就可以。  
接下来做两个事情，一个是通过拼接图片地址并且下载图片，二是通过组名来命名文件夹名称。然后把图片保存到相应的文件夹。代码如下。

```
\# 使用enumerate函数,取出列表元素值和下标
            for index, picture\_url\_tem in enumerate(url\_detail\_list):
                # 拼接图片地址
                picture\_url = self.url\_main+picture\_url\_tem
                # requests get 请求
                picture\_reponse = requests.get(picture\_url,headers=self.user\_headers)
                # 文件夹名字
                dir\_name = str(groupname\[0\])
                file\_name = str(index)+'.jpeg'

                if not os.path.exists(dir\_name):  # 判断文件夹是否存在，如果不存在：
                    os.mkdir(dir\_name)  # 创建一个文件夹

                with open(dir\_name + '/' + file\_name, 'wb') as f:  # 用wb模式打开创建文件，w写模式
                    f.write(picture\_reponse.content)  # 写入二进制文件内容
                time.sleep(5)
```

到此为止就把整个爬取流程写完了。把这些定义的方法，放到 run 方法跑起来。

```
def run(self):
        # 获取网易内容
        html\_main = self.get\_html\_main()
        # 获取详情页内容,筛选出每组图详情页网址
        detail\_url\_list = self.get\_detail\_url(html\_main)
        # 获取图片组名称，图片下载链接，下载并保存图片
        get\_pictures = self.get\_pictures\_groupname(detail\_url\_list)
```

最后放到 main 函数跑起来。

```
if \_\_name\_\_ == "\_\_main\_\_":
    # 实例化这个WeiMeiGirls类
    weimei = WeiMeiGirls()
    # 调用run方法
    weimei.run()
```

跑起来后，就吹着空调，挖着，撸撸猫，坐等收图。

*   ![](http://upload-images.jianshu.io/upload_images/1957277-24ca567021e96b2e.png)
    
    image.png
    
*   ![](http://upload-images.jianshu.io/upload_images/1957277-95eb353548e36c31.png)
    
    image.png
    

这里还有很多没有做好的，比如多线程，比如把这些获取的信息存入数据库等等。后面将继续优化。请关注下一篇文章。  
源码地址：[https://github.com/LesterZoeyXu/pachong](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2FLesterZoeyXu%2Fpachong)  
或者对 Python 感兴趣的朋友可以关注我的简书和公众号。  

![](http://upload-images.jianshu.io/upload_images/1957277-d2eb786d8e447403.jpg)

码农不头秃

"小礼物走一走，来简书关注我"

还没有人赞赏，支持一下

[![](https://upload.jianshu.io/users/upload_avatars/1957277/1a1cb45a-8012-4ab9-ad3f-3ad04d6b29a3.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/100/h/100/format/webp)](https://www.jianshu.com/u/12648978d547)

总资产 94 (约 6.35 元) 共写了 1.0W 字获得 22 个赞共 33 个粉丝

### 被以下专题收入，发现更多相似内容

### 推荐阅读[更多精彩内容](https://www.jianshu.com/)

*   妹子图网站爬取 --- 前言 从今天开始就要撸起袖子，直接写 Python 爬虫了，学习语言最好的办法就是有目的的进行，所...
    
    [![](https://upload.jianshu.io/users/upload_avatars/17885815/0528b494-2ddc-4ff5-862d-4f4bfa5a7206.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)IT 派森](https://www.jianshu.com/u/49675a550afb)阅读 0
    
*   今天感恩节哎，感谢一直在我身边的亲朋好友。感恩相遇！感恩不离不弃。 中午开了第一次的党会，身份的转变要...
    
    [![](https://upload-images.jianshu.io/upload_images/20029397-1d19b7dac95600da.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/300/h/240/format/webp)](https://www.jianshu.com/p/19f47749a042)