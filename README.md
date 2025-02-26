# 车牌识别系统

### 介绍
这是华南师范大学，阿伯丁数据科学与人工智能学院开展的数字图像处理课程的期末考核项目

基于Python，opencv库的车牌识别系统，纯图像处理，不含机器学习等方法。

**作者：** 陈宇宸

**专业：** 软件工程

**学号：** 20223802061

**邮箱：** 3037857646@qq.com / cyc20040505@gmain.com

### 目录结构
```
code/
├──dataset/
│   ├── hk/
│   └── inland/
├── template/
├── config.js
├── predict.py
├── surface.py
└── README.md
```

### 文件说明
- `code/`: 存放所有项目相关的数据文件，包括训练和测试数据，算法和UI界面代码。
- `dataset/`: 存放图片数据，用于提供测试项目运行的原图像。
- `hk/`：港澳车牌的图像。
- `inland/`：内地车牌的图像。
- `template/`: 存放用于识别的车牌字符模板，包括数字、英文及中文字符模板。
- `config.js`: 配置文件（JS格式）。
- `predict.py`: 车牌识别系统算法的代码。
- `surface.py`: UI界面的代码。
- `README.md`: 项目说明文件，提供项目概览、使用说明等信息。

### 使用方法
运行环境：anaconda
安装python、numpy、opencv、PIL后在code目录下运行surface.py文件即可

**创建环境**
```
conda create -n image_process python=3.8
```
**激活环境**
```
conda activate image_process
```
**安装所需要的库**
```
pip install numpy
pip install opencv-python
pip install pillow
```
**切换目录**
```
cd path/to/directory/code
```
**运行项目**
```
python surface.py
```
### 算法
算法思想来自网上资源，先使用图像边缘和车牌颜色定位车牌，再识别字符。车牌定位、字符识别均在algorithm.py文件中。


### 注意事项
1. 由于训练样本有限，车牌字符识别可能存在误差。
2. 系统暂时不能自动区分大陆和港澳的车牌类型，请选择相应的算法来测试。
