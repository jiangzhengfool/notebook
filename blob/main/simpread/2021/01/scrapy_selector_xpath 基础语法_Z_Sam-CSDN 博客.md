> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/u010553139/article/details/104006117)

XPath 表达式

<table><thead><tr><th>表达式</th><th>描述</th></tr></thead><tbody><tr><td>/</td><td>选中文档的根（root）</td></tr><tr><td>.</td><td>选中当前节点</td></tr><tr><td>…</td><td>选中当前节点的父节点</td></tr><tr><td>ELEMENT</td><td>选中子节点中所有 ELEMENT 元素节点</td></tr><tr><td>//ELEMENT</td><td>选中后代节点中所 ELEMENT 元素节点</td></tr><tr><td>*</td><td>选中所有元素子节点</td></tr><tr><td>text()</td><td>选中所文本子节点</td></tr><tr><td>@ATTR</td><td>选中名为 ATTR 的属性节点</td></tr><tr><td>@*</td><td>选中所有属性节点</td></tr><tr><td>[谓语]</td><td>谓语用来查找特定的节点或者包含某个特定值的节点</td></tr></tbody></table>

```
# 《精通 scrapy 网络爬虫》第3章 第3节（即3.3）xpath 实例

from scrapy.selector import Selector
from scrapy.http import HtmlResponse

body = '''
<html>
	<head>
		<base href='http://example.com'/>
		<title>Example website</title>
	</head>
	<body>
		<div id='images'>
			<a href='image1.html'>Name:Image 1 <br/><img src="image1.jpg"/></a>
			<a href='image2.html'>Name:Image 2 <br/><img src="image2.jpg"/></a>
			<a href='image3.html'>Name:Image 3 <br/><img src="image3.jpg"/></a>
			<a href='image4.html'>Name:Image 4 <br/><img src="image4.jpg"/></a>
			<a href='image5.html'>Name:Image 5 <br/><img src="image5.jpg"/></a>
		</div>
	</body>
</html>
'''
response = HtmlResponse(url='http://www.example.com/', body=body, encoding='utf-8')

# /: 一个从根开始的绝对路径
print('[1]==========/: 一个从根开始的绝对路径==========')
print(response.xpath('/html'))
print(response.xpath('/html/head'))
# E1/E2：选中E1节点中所有E2
print('[2]==========E1/E2：选中E1节点中所有E2==========')
print(response.xpath('/html/body/div/a'))
# //E: 选中文档中所有E，无论在什么位置
print('[3]==========//E: 选中文档中所有E，无论在什么位置==========')
print(response.xpath('//a'))
# E1//E2:选中E1后代节点中所有E2，无论在后代中的什么位置
print('[4]==========E1//E2:选中E1后代节点中所有E2，无论在后代中的什么位置==========')
print(response.xpath('/html/body//img'))
print(response.xpath('/html/body//a'))
# E/text():选中E的文本子节点
print('[5]==========E/text():选中E的文本子节点==========')
print(response.xpath('//a/text()'))
print(response.xpath('//a/text()'))
# E/*:选中E的所有元素子节点
print('[6]==========E/*:选中E的所有元素子节点==========')
print(response.xpath('/html/*'))
print(response.xpath('//body/*'))
# */E:选中孙节点中的所有E
print('[7]==========*/E:选中孙节点中的所有E==========')
print(response.xpath('//div/*/img'))
# E/@ATTR: 选中E的ATTR属性
print('[8]==========E/@ATTR: 选中E的ATTR属性==========')
print(response.xpath('//img/@src'))
# //@ATTR: 选中文档中所有Attr属性
print('[9]==========//@ATTR: 选中文档中所有Attr属性==========')
print(response.xpath('//@href'))
print(response.xpath('//@href'))
# E/@*:选中E的所有属性
print('[10]==========E/@*:选中E的所有属性==========')
print(response.xpath('//a[1]/img/@*'))
# .:选中当前节点，用来描述相对路径
print('[11]==========.:选中当前节点，用来描述相对路径==========')
sel = response.xpath('//a')[0]
print(sel)
print(sel.xpath('//img')) # //img 是绝对路径，从根开始搜索，不是从当前a开始
print(sel.xpath('.//img'))# .//img 描述当前节点后代中所有img
# ..: 选中当前节点的父节点，用来描述相对路径
print('[12]==========..: 选中当前节点的父节点，用来描述相对路径==========')
print(response.xpath('..//img'))
# node[谓语]：用来查找某个特定的节点或者包含某个特定值的节点
# a中的第3 个
print('[13]==========a中的第 3 个==========')
print(response.xpath('//a[3]'))
# last函数,选中最后1个
print('[14]==========last函数,选中最后1个==========')
print(response.xpath('//a[last()]'))
# position函数，选中前3个
print('[15]==========position函数，选中前3个==========')
print(response.xpath('//a[position()<=2]'))
# 选中所有含有id属性的div
print('[16]==========选中所有含有id属性的div==========')
print(response.xpath('//div[@id]'))
# 选中所有含有id属性且值为"images"的div
print('[17]==========选中所有含有id属性且值为"images"的div==========')
print(response.xpath('//div[@id="images"]'),'\n')

# 常用函数
print('==========xpath常用函数==========')
print('[1]==========string(arg):返回参数字符串值==========')
text = '<a href="#">Click here to go the <strong> Next Page </strong></a>'
sel = Selector(text=text)
print(sel)
print(sel.xpath('//a/strong').extract())
print(sel.xpath('string(/html/body/a/strong/text())').extract())
print(sel.xpath('/html/body/a//text()').extract())
print(sel.xpath('string(/html/body/a)').extract())
print('[2]==========contains(str1,str2):判断str1中是否包含str2，返回布尔值==========')
text = '''
<div>
    <p class="small info">hello world</p>
    <p class="normal info">hell scrapy</p>
</div>
'''
sel=Selector(text=text)
print(sel.xpath('//p'))
print(sel.xpath('//p[contains(@class,"small")]'))
print(sel.xpath('//p[contains(@class,"info")]'))
print(sel.xpath('//p[contains(@class,"info1")]'))
------------------------
D:\Python38\python.exe D:/Project0611/ScrapyBook/practise/scrapySelectorXpathTest.py
[1]==========/: 一个从根开始的绝对路径==========
[<Selector xpath='/html' data='<html>\n\t<head>\n\t\t<base href="http://e...'>]
[<Selector xpath='/html/head' data='<head>\n\t\t<base href="http://example.c...'>]
[2]==========E1/E2：选中E1节点中所有E2==========
[<Selector xpath='/html/body/div/a' data='<a href="image1.html">Name:Image 1 <b...'>, <Selector xpath='/html/body/div/a' data='<a href="image2.html">Name:Image 2 <b...'>, <Selector xpath='/html/body/div/a' data='<a href="image3.html">Name:Image 3 <b...'>, <Selector xpath='/html/body/div/a' data='<a href="image4.html">Name:Image 4 <b...'>, <Selector xpath='/html/body/div/a' data='<a href="image5.html">Name:Image 5 <b...'>]
[3]==========//E: 选中文档中所有E，无论在什么位置==========
[<Selector xpath='//a' data='<a href="image1.html">Name:Image 1 <b...'>, <Selector xpath='//a' data='<a href="image2.html">Name:Image 2 <b...'>, <Selector xpath='//a' data='<a href="image3.html">Name:Image 3 <b...'>, <Selector xpath='//a' data='<a href="image4.html">Name:Image 4 <b...'>, <Selector xpath='//a' data='<a href="image5.html">Name:Image 5 <b...'>]
[4]==========E1//E2:选中E1后代节点中所有E2，无论在后代中的什么位置==========
[<Selector xpath='/html/body//img' data='<img src="image1.jpg">'>, <Selector xpath='/html/body//img' data='<img src="image2.jpg">'>, <Selector xpath='/html/body//img' data='<img src="image3.jpg">'>, <Selector xpath='/html/body//img' data='<img src="image4.jpg">'>, <Selector xpath='/html/body//img' data='<img src="image5.jpg">'>]
[<Selector xpath='/html/body//a' data='<a href="image1.html">Name:Image 1 <b...'>, <Selector xpath='/html/body//a' data='<a href="image2.html">Name:Image 2 <b...'>, <Selector xpath='/html/body//a' data='<a href="image3.html">Name:Image 3 <b...'>, <Selector xpath='/html/body//a' data='<a href="image4.html">Name:Image 4 <b...'>, <Selector xpath='/html/body//a' data='<a href="image5.html">Name:Image 5 <b...'>]
[5]==========E/text():选中E的文本子节点==========
[<Selector xpath='//a/text()' data='Name:Image 1 '>, <Selector xpath='//a/text()' data='Name:Image 2 '>, <Selector xpath='//a/text()' data='Name:Image 3 '>, <Selector xpath='//a/text()' data='Name:Image 4 '>, <Selector xpath='//a/text()' data='Name:Image 5 '>]
[<Selector xpath='//a/text()' data='Name:Image 1 '>, <Selector xpath='//a/text()' data='Name:Image 2 '>, <Selector xpath='//a/text()' data='Name:Image 3 '>, <Selector xpath='//a/text()' data='Name:Image 4 '>, <Selector xpath='//a/text()' data='Name:Image 5 '>]
[6]==========E/*:选中E的所有元素子节点==========
[<Selector xpath='/html/*' data='<head>\n\t\t<base href="http://example.c...'>, <Selector xpath='/html/*' data='<body>\n\t\t<div id="images">\n\t\t\t<a href...'>]
[<Selector xpath='//body/*' data='<div id="images">\n\t\t\t<a href="image1....'>]
[7]==========*/E:选中孙节点中的所有E==========
[<Selector xpath='//div/*/img' data='<img src="image1.jpg">'>, <Selector xpath='//div/*/img' data='<img src="image2.jpg">'>, <Selector xpath='//div/*/img' data='<img src="image3.jpg">'>, <Selector xpath='//div/*/img' data='<img src="image4.jpg">'>, <Selector xpath='//div/*/img' data='<img src="image5.jpg">'>]
[8]==========E/@ATTR: 选中E的ATTR属性==========
[<Selector xpath='//img/@src' data='image1.jpg'>, <Selector xpath='//img/@src' data='image2.jpg'>, <Selector xpath='//img/@src' data='image3.jpg'>, <Selector xpath='//img/@src' data='image4.jpg'>, <Selector xpath='//img/@src' data='image5.jpg'>]
[9]==========//@ATTR: 选中文档中所有Attr属性==========
[<Selector xpath='//@href' data='http://example.com'>, <Selector xpath='//@href' data='image1.html'>, <Selector xpath='//@href' data='image2.html'>, <Selector xpath='//@href' data='image3.html'>, <Selector xpath='//@href' data='image4.html'>, <Selector xpath='//@href' data='image5.html'>]
[<Selector xpath='//@href' data='http://example.com'>, <Selector xpath='//@href' data='image1.html'>, <Selector xpath='//@href' data='image2.html'>, <Selector xpath='//@href' data='image3.html'>, <Selector xpath='//@href' data='image4.html'>, <Selector xpath='//@href' data='image5.html'>]
[10]==========E/@*:选中E的所有属性==========
[<Selector xpath='//a[1]/img/@*' data='image1.jpg'>]
[11]==========.:选中当前节点，用来描述相对路径==========
<Selector xpath='//a' data='<a href="image1.html">Name:Image 1 <b...'>
[<Selector xpath='//img' data='<img src="image1.jpg">'>, <Selector xpath='//img' data='<img src="image2.jpg">'>, <Selector xpath='//img' data='<img src="image3.jpg">'>, <Selector xpath='//img' data='<img src="image4.jpg">'>, <Selector xpath='//img' data='<img src="image5.jpg">'>]
[<Selector xpath='.//img' data='<img src="image1.jpg">'>]
[12]==========..: 选中当前节点的父节点，用来描述相对路径==========
[<Selector xpath='..//img' data='<img src="image1.jpg">'>, <Selector xpath='..//img' data='<img src="image2.jpg">'>, <Selector xpath='..//img' data='<img src="image3.jpg">'>, <Selector xpath='..//img' data='<img src="image4.jpg">'>, <Selector xpath='..//img' data='<img src="image5.jpg">'>]
[13]==========a中的第 3 个==========
[<Selector xpath='//a[3]' data='<a href="image3.html">Name:Image 3 <b...'>]
[14]==========last函数,选中最后1个==========
[<Selector xpath='//a[last()]' data='<a href="image5.html">Name:Image 5 <b...'>]
[15]==========position函数，选中前3个==========
[<Selector xpath='//a[position()<=2]' data='<a href="image1.html">Name:Image 1 <b...'>, <Selector xpath='//a[position()<=2]' data='<a href="image2.html">Name:Image 2 <b...'>]
[16]==========选中所有含有id属性的div==========
[<Selector xpath='//div[@id]' data='<div id="images">\n\t\t\t<a href="image1....'>]
[17]==========选中所有含有id属性且值为"images"的div==========
[<Selector xpath='//div[@id="images"]' data='<div id="images">\n\t\t\t<a href="image1....'>] 

==========xpath常用函数==========
[1]==========string(arg):返回参数字符串值==========
<Selector xpath=None data='<html><body><a href="#">Click here to...'>
['<strong> Next Page </strong>']
[' Next Page ']
['Click here to go the ', ' Next Page ']
['Click here to go the  Next Page ']
[2]==========contains(str1,str2):判断str1中是否包含str2，返回布尔值==========
[<Selector xpath='//p' data='<p class="small info">hello world</p>'>, <Selector xpath='//p' data='<p class="normal info">hell scrapy</p>'>]
[<Selector xpath='//p[contains(@class,"small")]' data='<p class="small info">hello world</p>'>]
[<Selector xpath='//p[contains(@class,"info")]' data='<p class="small info">hello world</p>'>, <Selector xpath='//p[contains(@class,"info")]' data='<p class="normal info">hell scrapy</p>'>]
[]

Process finished with exit code 0


```

[更多代码](https://github.com/zkzhang1986/-Scrapy-)