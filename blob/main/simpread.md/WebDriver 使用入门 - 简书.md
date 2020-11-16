\> 本文由 \[简悦 SimpRead\](http://ksria.com/simpread/) 转码， 原文地址 \[www.jianshu.com\](https://www.jianshu.com/p/db983d0ea6fa)

[![](https://upload.jianshu.io/users/upload_avatars/6218810/f65d4610-596c-42dd-87b4-9a53738d3bc3.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/96/h/96/format/webp)](https://www.jianshu.com/u/a5c7473beab9)

2017.12.29 14:31:46 字数 1,359 阅读 2,326

WD.py 是一个 Python WebDriver 客户端，实现了 WebDriver 协议中的大部分 API。 它最初是为 Macaca（一个 Node.js 驱动的 WebDriver 服务器）而设计的，但也可以应用于 WebDriver 服务器的其他实现，比如 Selenium，Appium 等。

![](http://upload-images.jianshu.io/upload_images/6218810-8ef7cd45246dc3a6.jpg)

自动化测试

查找元素
----

WebDriver 的查找元素命令允许分别查找单个元素和元素集合，定位策略如下：

*   文本
*   id
*   XPath
*   链接文本
*   部分链接文本
*   标签名
*   类名
*   CSS 选择器

找到元素的基本方法是`element()`，例如查找 id 为 “login” 的元素：

```
driver.element('id', 'login')
```

但是在大多数情况下，不需要使用这个基本的方法，为了方便起见，有很多扩展方法。

### 文本

例如，查找页面上 name 属性为 “Login failed!” 的元素：

```
driver.element\_by\_name('Login failed!')
```

### id

例如，查找 id 为 “login” 的元素：

```
driver.element\_by\_id('login')
```

### XPath

XPath 是 XML Path 的简称，由于 HTML 文档本身就是一个标准的 XML 页面，所以可以使用 XPath 的语法来定位页面元素。这个方法是非常强大的元素查找方式，使用这种方法几乎可以定位到页面上的任意元素。

例如，查找页面上 id 为 “finding-elements-to-interact” 的元素下的第 4 个 table 元素：

```
driver.element\_by\_xpath('//\*\[@id="finding-elements-to-interact"\]/table\[4\]')
```

### 链接文本

这个方法比较直接，即通过超文本链接上的文字信息来定位元素，这种方式一般专门用于定位页面上的超文本链接。

例如，查找页面上文字为 “macaca” 的超文本链接：

```
driver.element\_by\_link\_text('macaca')
```

### 部分链接文本

这个方法是上一个方法的扩展，当不能准确知道超链接上的文本信息或者只想通过一些关键字进行匹配时，可以使用这个方法来通过部分链接文字进行匹配。

例如，查找页面上部分文字为 “maca” 的超文本链接：

```
driver.element\_by\_partial\_link\_text('maca')
```

### 标签名

该方法可以通过元素的标签名称来查找元素，需要注意的是，这个方法搜索到的元素通常不止一个。

例如，查找页面上的 “input” 标签：

```
driver.element\_by\_tag\_name('input')
```

此外，WebDriver 上的所有元素方法都可以在 WebElement 上使用，也就是从当前 Web 元素中查找元素：

```
web\_element.element\_by\_id('ss')
```

### 类名

一般程序员或页面设计师会给元素直接赋予一个样式属性或者利用 css 文件里的伪类来定义元素样式，这个方法可以利用元素的 css 样式表所引用的伪类名称来进行元素查找。

例如，查找页面上 className 属性为 “btn” 的元素：

```
driver.element\_by\_class\_name('btn')
```

### CSS 选择器

这种元素定位方式跟 XPath 比较类似，但执行速度较快，所以功能也是蛮强大的。

例如，查找页面上样式表为 “.btn” 的元素：

```
driver.element\_by\_css\_selector('.btn')
```

### 异常处理

当没有找到元素时，会引发`WebDriverException`异常。为了避免这种情况，可是使用`element_if_exists`方法，如果元素存在，则返回 True，否则返回 False，例如：

```
driver.element\_by\_id\_if\_exists('login')
```

也可以使用`element_or_none`方法，如果元素存在，则返回元素，否则返回 None，例如：

```
driver.element\_by\_id\_or\_none('login')
```

此外，还有`wait_for`方法等待元素满足给定条件，默认等待 10 秒，每个间隔 1 秒，断言器函数默认为`asserters.is_displayed`，例如：

```
driver.wait\_for\_element\_by\_id('login')
```

按键输入
----

当需要完成一个输入字段的操作时，可以将一系列的按键行为发送给一个元素：

```
driver.web\_element.send\_keys('123456')
```

`send_keys`方法也接受一个数组，这在发送特殊键（不是文本的按键）时非常有用：

```
driver.web\_element.send\_keys(\[1, 2, 3, 4, 5, 6\])
```

PC 按键映射：

<table><thead><tr><th>映射</th><th>按键</th><th>键码</th></tr></thead><tbody><tr><td>\uE002</td><td>HELP</td><td>259 (0x00000103)</td></tr><tr><td>\uE003</td><td>BACK_SPACE</td><td>67 (0x00000043)</td></tr><tr><td>\uE004</td><td>TAB</td><td>61 (0x0000003d)</td></tr><tr><td>\uE005</td><td>CLEAR</td><td>28 (0x0000001c)</td></tr><tr><td>\uE007</td><td>ENTER</td><td>66 (0x00000042)</td></tr><tr><td>\uE008</td><td>SHIFT</td><td>59 (0x0000003b)</td></tr><tr><td>\uE009</td><td>CONTROL</td><td>113 (0x00000071)</td></tr><tr><td>\uE00A</td><td>ALT</td><td>57 (0x00000039)</td></tr><tr><td>\uE00B</td><td>PAUSE</td><td>121 (0x00000079)</td></tr><tr><td>\uE00C</td><td>ESCAPE</td><td>111 (0x0000006f)</td></tr><tr><td>\uE00E</td><td>PAGE_UP</td><td>92 (0x0000005c)</td></tr><tr><td>\uE00F</td><td>PAGE_DOWN</td><td>93 (0x0000005d)</td></tr><tr><td>\uE010</td><td>END</td><td>123 (0x0000007b)</td></tr><tr><td>\uE011</td><td>HOME</td><td>122 (0x0000007a)</td></tr><tr><td>\uE012</td><td>ARROW_LEFT</td><td>21 (0x00000015)</td></tr><tr><td>\uE013</td><td>ARROW_UP</td><td>19 (0x00000013)</td></tr><tr><td>\uE014</td><td>ARROW_RIGHT</td><td>22 (0x00000016)</td></tr><tr><td>\uE015</td><td>ARROW_DOWN</td><td>20 (0x00000014)</td></tr><tr><td>\uE016</td><td>INSERT</td><td>124 (0x0000007c)</td></tr><tr><td>\uE017</td><td>DELETE</td><td>112 (0x00000070)</td></tr><tr><td>\uE031</td><td>F1</td><td>131 (0x00000083)</td></tr><tr><td>\uE032</td><td>F2</td><td>132 (0x00000084)</td></tr><tr><td>\uE033</td><td>F3</td><td>133 (0x00000085)</td></tr><tr><td>\uE034</td><td>F4</td><td>134 (0x00000086)</td></tr><tr><td>\uE035</td><td>F5</td><td>135 (0x00000087)</td></tr><tr><td>\uE036</td><td>F6</td><td>136 (0x00000088)</td></tr><tr><td>\uE037</td><td>F7</td><td>137 (0x00000089)</td></tr><tr><td>\uE038</td><td>F8</td><td>138 (0x0000008a)</td></tr><tr><td>\uE039</td><td>F9</td><td>139 (0x0000008b)</td></tr><tr><td>\uE03A</td><td>F10</td><td>140 (0x0000008c)</td></tr><tr><td>\uE03B</td><td>F11</td><td>141 (0x0000008d)</td></tr><tr><td>\uE03C</td><td>F12</td><td>142 (0x0000008e)</td></tr><tr><td>\uE03D</td><td>META</td><td>117 (0x00000075)</td></tr></tbody></table>

Android 按键映射：

<table><thead><tr><th>映射</th><th>按键</th><th>键码</th></tr></thead><tbody><tr><td>\uE101</td><td>POWER 电源键</td><td>26 (0x0000001a)</td></tr><tr><td>\uE102</td><td>VOLUME_UP 音量加</td><td>24 (0x00000018)</td></tr><tr><td>\uE103</td><td>VOLUME_DOWN 音量减</td><td>25 (0x00000019)</td></tr><tr><td>\uE104</td><td>VOLUME_MUTE 禁音</td><td>164 (0x000000a4)</td></tr><tr><td>\uE105</td><td>HOME_SCREEN HOME 键</td><td>3 (0x00000003)</td></tr><tr><td>\uE106</td><td>BACK BACK 键</td><td>4 (0x00000004)</td></tr><tr><td>\uE107</td><td>MENU MENU 键</td><td>82 (0x00000052)</td></tr><tr><td>\uE108</td><td>CAMERA 拍照键</td><td>27 (0x0000001b)</td></tr><tr><td>\uE109</td><td>CALL 电话键</td><td>5 (0x00000005)</td></tr><tr><td>\uE10A</td><td>END_CALL 结束电话键</td><td>6 (0x00000006)</td></tr><tr><td>\uE10B</td><td>SEARCH 搜索键</td><td>84 (0x00000054)</td></tr><tr><td>\uE10C</td><td>DPAD_LEFT 导航左键</td><td>21 (0x00000015)</td></tr><tr><td>\uE10D</td><td>DPAD_UP 导航上键</td><td>19 (0x00000013)</td></tr><tr><td>\uE10E</td><td>DPAD_RIGHT 导航右键</td><td>22 (0x00000016)</td></tr><tr><td>\uE10F</td><td>DPAD_DOWN 导航下键</td><td>20 (0x00000014)</td></tr><tr><td>\uE110</td><td>DPAD_CENTER 导航确定键</td><td>23 (0x00000017)</td></tr></tbody></table>

iOS 按键映射：

<table><thead><tr><th>映射</th><th>按键</th></tr></thead><tbody><tr><td>\uE105</td><td>HOME_SCREEN HOME 键</td></tr></tbody></table>

使用数组发送特殊键非常方便：

```
driver.web\_element.send\_keys(\[1, DELETE, 1, 2, 3, 4, 5, 6\])
```

屏幕快照
----

截图时可以返回截图的`base64`编码字符串：

```
base64\_str = driver.take\_screenshot()
```

或者保存截图到给定的路径：

```
driver.save\_screenshot('./screen.png')
```

`save_screenshot`方法具有可选的第二个参数来决定是否由于某种原因无法保存到文件系统时忽略`IOError`。例如，没有读写权限时，忽略异常信息：

```
driver.save\_screenshot('/etc/screen.png', True)
```

切换环境
----

对于移动端测试，可能需要在 Native（原生）和 Webview（H5）之间切换环境，首先获取现有的环境：

```
ctxs = driver.contexts
print(ctxs) # \['NATIVE', 'WEBVIEW\_1', 'WEBVIEW\_2'\]
```

然后切换到指定的环境：

```
driver.context = 'WEBVIEW\_1'
print(driver.context) # WEBVIEW\_1
```

执行 JS 片段
--------

在一些复杂的情况下，可能需要在页面中插入一段 JavaScript 代码，并得到想要的任何东西。可以在脚本中使用`arguments`来表示脚本之后的索引参数。

```
script = 'return document.querySelector(".btn").tagName === arguments\[0\]'
args = \['div'\]
result = driver.execute\_script(script, \*args)
```

上面的脚本等于 JavaScript 中的 IIFE：

```
function () {
  return document.querySelector(".btn").tagName === "div"
}()
```

WebElement 方法
-------------

WebElement 的实例方法主要与行为元素有关，比如点击元素、获取标签名或元素的内部文本。WebElement 实例是通过查找元素命令返回的，例如通过 id 检索元素：

```
web\_element = driver.element\_by\_id('login')
print(type(web\_element) == WebElement) # True
```

例如点击元素：

例如获取元素的标签名称：

```
web\_element.click()
```

例如获取元素的内部文本：

编码风格
----

建议使用官方推荐的两种编码风格，第一种是使用额外的括号：

```
tag\_name = web\_element.tag\_name
```

第二种是使用反斜杠：

```
text = web\_element.text
```

"小礼物走一走，来简书关注我"

还没有人赞赏，支持一下

[![](https://upload.jianshu.io/users/upload_avatars/6218810/f65d4610-596c-42dd-87b4-9a53738d3bc3.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/100/h/100/format/webp)](https://www.jianshu.com/u/a5c7473beab9)

总资产 49 共写了 8.8W 字获得 371 个赞共 324 个粉丝

### 被以下专题收入，发现更多相似内容

### 推荐阅读[更多精彩内容](https://www.jianshu.com/)

*   Selenium 官网 Selenium WebDriver 官网 webdriver 实用指南 python 版本 WebD...
    
    [![](https://upload.jianshu.io/users/upload_avatars/3971414/a33cd710-df92-4081-a750-38545c9876fd.jpeg?imageMogr2/auto-orient/strip|imageView2/1/w/48/h/48/format/webp)顾顾 314](https://www.jianshu.com/u/6da1e5938717) 阅读 3
    
    [![](https://upload-images.jianshu.io/upload_images/2286765-0b311e893a4ddc38.png?imageMogr2/auto-orient/strip|imageView2/1/w/300/h/240/format/webp)](https://www.jianshu.com/p/9f563b89f086)
*   WebDriver 进阶 欢迎阅读 WebDriver 进阶讲义。本篇讲义将会重点介绍 Selenium WebDriv...
    
*   洁净和食物 在开始练习瑜伽体式前，应该先排空膀胱，清空肠胃。一些倒立体式有助于膀胱的活动，假如练习者患...
    
    [![](https://upload-images.jianshu.io/upload_images/11246061-fd5dafd36e403f71.jpg?imageMogr2/auto-orient/strip|imageView2/1/w/300/h/240/format/webp)](https://www.jianshu.com/p/bb623d138566)