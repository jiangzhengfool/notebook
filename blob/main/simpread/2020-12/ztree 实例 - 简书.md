> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.jianshu.com](https://www.jianshu.com/p/38b3aa848f44)

[转自：](https://link.jianshu.com?t=%255Bhttp%3A%2F%2Fblog.csdn.net%2Fduck_arrow%2Farticle%2Fdetails%2F7091861%255D%28http%3A%2F%2Fblog.csdn.net%2Fduck_arrow%2Farticle%2Fdetails%2F7091861%29)  
zTree【简介】  
zTree 是利用 [jQuery](https://link.jianshu.com?t=http%3A%2F%2Flib.csdn.net%2Fbase%2Fjquery) 的核心代码，实现一套能完成大部分常用功能的 Tree 插件  
兼容 IE、FireFox、Chrome 等浏览器  
在一个页面内可同时生成多个 Tree 实例  
支持 JSON 数据  
支持一次性静态生成 和 Ajax 异步加载 两种方式  
支持多种事件响应及反馈  
支持 Tree 的节点移动、编辑、删除  
支持任意更换皮肤 / 个性化图标（依靠 css）  
支持极其灵活的 checkbox 或 radio 选择功能  
简单的参数配置实现 灵活多变的功能

【官网】  
[官网地址：](https://link.jianshu.com?t=http%3A%2F%2Fwww.baby666.cn%2Fhunter%2Findex.html)  
在官网能够下载到 zTree 的源码、实例和 API，其中作者 pdf 的 API 写得非常详细  
【部分函数和属性介绍】  
核心：zTree(setting, [zTreeNodes])

这个函数接受一个 JSON 格式的数据对象 setting 和一个 JSON 格式的数据对象 zTreeNodes，从而建立 Tree。

##### 核心参数：setting

zTree 的参数配置都在这里完成，简单的说：树的样式、事件、访问路径等都在这里配置  
setting 举例:

```
var setting = { 
showLine: true, 
checkable: true 
};
```

因为参数太多，具体参数详见 zTreeAPI  
核心参数：zTreeNodes  
zTree 的全部节点数据集合，采用由 JSON 对象组成的[数据结构](https://link.jianshu.com?t=http%3A%2F%2Flib.csdn.net%2Fbase%2Fdatastructure)，简单的说：这里使用 Json 格式保存了树的所有信息  
zTreeNodes 的格式分为两种：利用 Json 格式嵌套体现父子关系和 Array 简单格式  
1, 带有父子关系的标准 zTreeNodes 举例:

```
var zTreeNodes = [ 
{"id":1, "name":"test1", "nodes":[ 
{"id":11, "name":"test11", "nodes":[ 
{"id":111, "name":"test111"} 
]}, 
{"id":12, "name":"test12"} 
]}, 
...... 
];
```

2 , 带有父子关系的简单 Array 格式 (isSimpleData) 的 zTreeNodes 举例：

```
var treeNodes = [ 
{"id":1, "pId":0, "name":"test1"}, 
{"id":11, "pId":1, "name":"test11"}, 
{"id":12, "pId":1, "name":"test12"}, 
{"id":111, "pId":11, "name":"test111"}, 
...... 
];
```

【实例一】([Java](https://link.jianshu.com?t=http%3A%2F%2Flib.csdn.net%2Fbase%2Fjavase) 代码)  
1. 在页面引用 zTree 的 js 和 css：

```
<!-- ZTree树形插件 -->
<link rel="stylesheet" href="<%=root%>/Web/common/css/zTreeStyle/zTreeStyle.css" type="text/css">
<!-- <link rel="stylesheet" href="<%=root%>/Web/common/css/zTreeStyle/zTreeIcons.css" type="text/css"> -->
<script type="text/[JavaScript](http://lib.csdn.net/base/javascript "JavaScript知识库")" src="<%=root%>/Web/common/js/jquery-ztree-2.5.min.js"></script>
```

2. 在 script 脚本中定义 setting 和 zTreeNodes

```
var setting = {
isSimpleData : true, //数据是否采用简单 Array 格式，默认false
treeNodeKey : "id", //在isSimpleData格式下，当前节点id属性
treeNodeParentKey : "pId", //在isSimpleData格式下，当前节点的父节点id属性
showLine : true, //是否显示节点间的连线
checkable : true //每个节点上是否显示 CheckBox
};
var treeNodes = [ 
{"id":1, "pId":0, "name":"test1"}, 
{"id":11, "pId":1, "name":"test11"}, 
{"id":12, "pId":1, "name":"test12"}, 
{"id":111, "pId":11, "name":"test111"}, 
];
```

3. 在进入页面时生成树结构：

```
var zTree;
$(function() {
zTree = $("#tree").zTree(setting, treeNodes);
});
```

4. 最后查看效果：

==========  
【实例二】(从后台获取简单格式 Json 数据)

1.  后台代码封装简单格式 Json 数据：

```
public void doGetPrivilegeTree() throws IOException{
String s1 = "{id:1, pId:0, name:\"test1\" , open:true}";
String s2 = "{id:2, pId:1, name:\"test2\" , open:true}";
String s3 = "{id:3, pId:1, name:\"test3\" , open:true}";
String s4 = "{id:4, pId:2, name:\"test4\" , open:true}";
List<String> lstTree = new ArrayList<String>();
lstTree.add(s1);
lstTree.add(s2);
lstTree.add(s3);
lstTree.add(s4);
//利用Json插件将Array转换成Json格式
response.getWriter().print(JSONArray.fromObject(lstTree).toString());
}
```

2.  页面使用 Ajax 获取 zTreeNodes 数据再生成树

```
var setting = {
isSimpleData : true, //数据是否采用简单 Array 格式，默认false
treeNodeKey : "id", //在isSimpleData格式下，当前节点id属性
treeNodeParentKey : "pId", //在isSimpleData格式下，当前节点的父节点id属性
showLine : true, //是否显示节点间的连线
checkable : true //每个节点上是否显示 CheckBox
};
var zTree;
var treeNodes;
$(function(){
$.ajax({
async : false,
cache:false,
type: 'POST',
dataType : "json",
url: root+"/ospm/loginInfo/doGetPrivilegeTree.action",//请求的action路径
error: function () {//请求失败处理函数
alert('请求失败');
},
success:function(data){ //请求成功后处理函数。 
alert(data);
treeNodes = data; //把后台封装好的简单Json格式赋给treeNodes
}
});
zTree = $("#tree").zTree(setting, treeNodes);
});
```

3.  最后显示效果

=======  
另外一个实例，转自：[http://blog.csdn.net/panpanhm91/article/details/7297130](https://link.jianshu.com?t=http%3A%2F%2Fblog.csdn.net%2Fpanpanhm91%2Farticle%2Fdetails%2F7297130)

首先下载 zTree 3.0 的 jar 包 (JQuery zTree v3.0.zip)，导入下面引用的相关 js

1、数据在页面获取。

```
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "[http://www.w3.org/TR/html4/loose.dtd](http://www.w3.org/TR/html4/loose.dtd)">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Insert title here</title>
<%
    String path = request.getContextPath();
   String basePath = request.getScheme() + "://"
     + request.getServerName() + ":" + request.getServerPort()
     + path + "/";
%>
<base href="<%=basePath%>">
<link rel="stylesheet" href="<%=basePath%>css/demo.css" type="text/css">
 <link rel="stylesheet" href="<%=basePath%>css/zTreeStyle/zTreeStyle.css" type="text/css">
 <script type="text/javascript" src="<%=basePath%>js/jquery-1.4.4.min.js"></script>
 <script type="text/javascript" src="<%=basePath%>js/jquery.ztree.core-3.0.js"></script>
 <script type="text/javascript" src="<%=basePath%>js/jquery.ztree.excheck-3.0.js"></script>
 <!--
 <script type="text/javascript" src="../../../js/jquery.ztree.exedit-3.0.js"></script>
 -->
 <SCRIPT type="text/javascript">
  <!--
  var setting = {
   check: {
    enable: true
   },
   data: {
    simpleData: {
     enable: true
    }
   }
  };

  var zNodes =[
   { id:1, pId:0, name:"随意勾选 1", open:true},
   { id:11, pId:1, name:"随意勾选 1-1", open:true},
   { id:111, pId:11, name:"随意勾选 1-1-1"},
   { id:112, pId:11, name:"随意勾选 1-1-2"},
   { id:12, pId:1, name:"随意勾选 1-2", open:true},
   { id:121, pId:12, name:"随意勾选 1-2-1"},
   { id:122, pId:12, name:"随意勾选 1-2-2"},
   { id:2, pId:0, name:"随意勾选 2", checked:true, open:true},
   { id:21, pId:2, name:"随意勾选 2-1"},
   { id:22, pId:2, name:"随意勾选 2-2", open:true},
   { id:221, pId:22, name:"随意勾选 2-2-1", checked:true},
   { id:222, pId:22, name:"随意勾选 2-2-2"},
   { id:23, pId:2, name:"随意勾选 2-3"}
  ];

  var code;

  function setCheck() {
   var zTree = $.fn.zTree.getZTreeObj("treeDemo"),
   py = $("#py").attr("checked")? "p":"",
   sy = $("#sy").attr("checked")? "s":"",
   pn = $("#pn").attr("checked")? "p":"",
   sn = $("#sn").attr("checked")? "s":"",
   type = { "Y":py + sy, "N":pn + sn};
   zTree.setting.check.chkboxType = type;
   showCode('setting.check.chkboxType = { "Y" : "' + type.Y + '", "N" : "' + type.N + '" };');
  }
  function showCode(str) {
   if (!code) code = $("#code");
   code.empty();
   code.append("<li>"+str+"</li>");
  }

  $(document).ready(function(){
   $.fn.zTree.init($("#treeDemo"), setting, zNodes);
   setCheck();
   $("#py").bind("change", setCheck);
   $("#sy").bind("change", setCheck);
   $("#pn").bind("change", setCheck);
   $("#sn").bind("change", setCheck);
  });
  //-->
 </SCRIPT>
</HEAD>

<BODY>
<h1>Checkbox 勾选操作</h1>
<h6></h6>
<div class="content_wrap">
 <div class="zTreeDemoBackground left">
  <ul id="treeDemo" class="ztree"></ul>
 </div>

</div>

</body>
</html>
```

2、数据从后台获取。

1）页面

```
<title>Insert title here</title>
<%
    String path = request.getContextPath();
   String basePath = request.getScheme() + "://"
     + request.getServerName() + ":" + request.getServerPort()
     + path + "/";
%>
<base href="<%=basePath%>">
<link rel="stylesheet" href="<%=basePath%>css/demo.css" type="text/css">
<link rel="stylesheet" href="<%=basePath%>css/zTreeStyle/zTreeStyle.css"
 type="text/css">
<script type="text/javascript" src="<%=basePath%>js/jquery-1.4.4.min.js"></script>
<script type="text/javascript"
 src="<%=basePath%>js/jquery.ztree.core-3.0.js"></script>
<script type="text/javascript"
 src="<%=basePath%>js/jquery.ztree.excheck-3.0.js"></script>
<SCRIPT type="text/javascript">

   var setting = {
    check: {
     enable: true
    },
    data: {
     simpleData: {
      enable: true
     }
    }
   }; 

 var zTree;
 var treeNodes;
 var code;

 $(function() {
  $.ajax({
   async : false,
   cache : false,
   type : "POST",
   dataType : "String",
   url : "menu.action",//恳求的action路径
   error : function() {//恳求失败处理惩罚函数
    alert("恳求失败");
   },
   success : function(data) { //恳求成功后处理惩罚函数。
    alert(data);

    // console.log(data); // 火狐在后台打印的日志。 
    treeNodes = data; //把后台封装好的简单Json格局赋给treeNodes
   }
  });

  //将string类型转换成json 
   treeNodes = eval("(" + treeNodes + ")");  

  zTree = $.fn.zTree.init($("#treeMenu"), setting, treeNodes);

 });  

</SCRIPT>
</HEAD>

<BODY>
 <h1>Checkbox 勾选操作</h1>
 <h6></h6>
 <div class="content_wrap">
  <div class="zTreeDemoBackground left">
   <ul id="treeMenu" class="ztree"></ul>
  </div>
 </div>
</body>
</html>
```

2）后台

```
/**
     * 
     * 查询所有菜单。
     * 
     * @return String
     */
    public String queryAllMenu()
    {
        List<String> lstTree = new ArrayList<String>();
        listMenu = this.menuService.queryMenu();
        HttpServletResponse response = ServletActionContext.getResponse();

        for (MenuInfo menu : listMenu)
        {
            String strMenu = null;

            strMenu = "{id:" + menu.getMenuId() + "," + "pId:"
                + menu.getParentMenuId() + "," + "name:"
                + "'"+menu.getMenuName()+"'" +","+ "open:true" + "}";

            lstTree.add(strMenu);
        }
        printWriter(response, JSONArray.fromObject(lstTree).toString());
        return "success";
    }

    /**
     * 
     * Description:将字符串写入Http响应 <br>
     * 
     * @param response
     *            response
     * @param outStr
     *            outStr
     * @see
     */
    private void printWriter(HttpServletResponse response, String outStr)
    {
        // 返回拼裝的数据
        response.setCharacterEncoding("UTF-8");
        response.setContentType("text/html");
        PrintWriter out = null;
        try
        {
            out = response.getWriter();
            out.print(outStr);
        }
        catch (IOException e)
        {
            DEBUGGER.error(e.toString());
        }
        finally
        {
            if (null != out)
            {
                out.close();
            }
        }
    }
```

=====  
另外一个实例，这个实例采用异步方式获取数据，转自：[http://blog.csdn.net/zyujie/article/details/7027663](https://link.jianshu.com?t=http%3A%2F%2Fblog.csdn.net%2Fzyujie%2Farticle%2Fdetails%2F7027663)

一个做. NET WEB 开发的朋友介绍了 ZTREE，它是基于 JQUERY 库开发的树型控件。于是去官方下了一个开发包，看了看 DEMO，觉得效果很不错，自己也做了个小例子，也许不太成形，效果倒是展现出来了，把使用方法记录下来，和大家分享分享。

1、新建了一个 HTML 在 <head> 标签内导入需要的 JS 和 CSS 文件。

```
<script language="javascript" type="text/javascript" src="js/jquery-1.6.4.js"></script>
<script language="javascript" type="text/javascript" src="js/jquery.ztree.core-3.0.js"></script>
<link type="text/css" rel="stylesheet" href="css/zTreeStyle/zTreeStyle.css" />
<script language="javascript" type="text/javascript" src="js/jquery.ztree.excheck-3.0.js"></script>
```

我这里使用的是 jquery1.6.4，jquery 的开发包这是必须的，然后导入 ztree.core 核心包，这里使用的是 ztree-3.0 还是 beta 版本的。呵呵，ztree.excheck-3.0 这是升级树控件，比如加上 checkbox 这些。

2、<script> 部分就直接贴代码了

```
<script type="text/javascript" language="javascript"> 
/**ztree的参数配置，setting主要是设置一些tree的属性，是本地数据源，还是远程，动画效果，是否含有复选框等等**/   
var setting = {
 check: { /**复选框**/
  enable: false,
  chkboxType: {"Y":"", "N":""}
 },
 view: {                                   
  //dblClickExpand: false,
  expandSpeed: 300 //设置树展开的动画速度，IE6下面没效果，
 },                           
 data: {                                   
  simpleData: {   //简单的数据源，一般开发中都是从[数据库](http://lib.csdn.net/base/mysql "MySQL知识库")里读取，API有介绍，这里只是本地的                         
   enable: true,
   idKey: "id",  //id和pid，这里不用多说了吧，树的目录级别
   pIdKey: "pId",
   rootPId: 0   //根节点
  }                           
 },                          
 callback: {     /**回调函数的设置，随便写了两个**/
  beforeClick: beforeClick,                                   
  onCheck: onCheck                           
 } 
};
function beforeClick(treeId, treeNode) {
 alert("beforeClick");
}
function onCheck(e, treeId, treeNode) {
 alert("onCheck");
}     

var citynodes = [      /**自定义的数据源，ztree支持json,数组，xml等格式的**/
 {id:0, pId:-1, name:"中国"},
 {id:1, pId:0, name:"北京"},  
 {id:2, pId:0, name:"天津"}, 
 {id:3, pId:0, name:"上海"},  
 {id:6, pId:0, name:"重庆"},  
 {id:4, pId:0, name:"河北省", open:false, nocheck:true},  
 {id:41, pId:4, name:"石家庄"},  
 {id:42, pId:4, name:"保定"},  
 {id:43, pId:4, name:"邯郸"},  
 {id:44, pId:4, name:"承德"},  
 {id:5, pId:0, name:"广东省", open:false, nocheck:true},  
 {id:51, pId:5, name:"广州"},  
 {id:52, pId:5, name:"深圳"},  
 {id:53, pId:5, name:"东莞"},  
 {id:54, pId:5, name:"佛山"},  
 {id:6, pId:0, name:"福建省", open:false, nocheck:true},  
 {id:61, pId:6, name:"福州"},  
 {id:62, pId:6, name:"厦门"},  
 {id:63, pId:6, name:"泉州"},  
 {id:64, pId:6, name:"三明"},
 {id:7, pId:0, name:"四川省", open:true, nocheck:true},
 {id:71, pId:7, name:"成都"},
 {id:72, pId:7, name:"绵阳"},
 {id:73, pId:7, name:"自贡"},
 {id:711, pId:71, name:"金牛区"},
 {id:712, pId:71, name:"锦江区"},
 {id:7111, pId:711, name:"九里堤"},
 {id:7112, pId:711, name:"火车北站"}
];

$(document).ready(function(){//初始化ztree对象    
  var zTreeDemo = $.fn.zTree.init($("#cityTree"),setting, citynodes);
});
</script>
```

3、body 部分，就一个

```
<ul id="cityTree" class="ztree"></ul>
```

4、当异步获取数据库的数据时，我们需要修改 setting 设置，也是返回的数组形式的数据：

```
var setting = {
 async: {   
        enable: true,   
        type:'post',
        [url:"treedata.jsp](http://blog.csdn.net/zyujie/article/details/'treedata.jsp)"
        ///dataFilter: filter
    },    
    data: {                                   
  simpleData: {   //简单的数据源，一般开发中都是从数据库里读取，API有介绍，这里只是本地的                          
   enable: true,
   idKey: "id",  //id和pid，这里不用多说了吧，树的目录级别
   pIdKey: "pId",
   rootPId: 0   //根节点
  }                           
 },                     
 callback: {     
  onAsyncSuccess: zTreeOnAsyncSuccess  /**回调函数的设置，异步提交成功的回调函数**/ 
 } 
};

$(document).ready(function(){//初始化ztree对象    
 $.fn.zTree.init($("#cityTree"), setting);  
});
```