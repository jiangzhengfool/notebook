> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [www.jianshu.com](https://www.jianshu.com/p/0ab80f63af8a)

*   安装  
    如果之前安装过显示目录功能的话，这一步骤可以跳过。  
    `pip install jupyter_contrib_nbextensions`
    
*   配置  
    安装完之后需要配置 nbextension，注意配置的时候要确保**已关闭** Jupyter Notebook：  
    `jupyter contrib nbextension install --user --skip-running-check`
    
*   启动 Jupyter Notebook，勾选设置  
    上面两个步骤都没报错后，启动 Jupyter Notebook，上面选项栏会出现 Nbextensions 的选项
    
    ![](http://upload-images.jianshu.io/upload_images/2759738-f0b422455e3d78b5.png)
    
    点开 Nbextensions 的选项，并勾选 Hinterland
    
    ![](http://upload-images.jianshu.io/upload_images/2759738-72952ade69a47155.png)
    
    使用效果：
    
    ![](http://upload-images.jianshu.io/upload_images/2759738-857586f61c00363a.jpg)

*   Jupyter Lab 中的自动补全功能  
    按 Tab 键即可使用。
    
    ![](http://upload-images.jianshu.io/upload_images/2759738-5da5313ea5f01167.jpg)