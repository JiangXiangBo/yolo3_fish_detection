# yolo3_fish_detection
基于yolo3的鱼和人脸的目标检测
## 百度链接：https://pan.baidu.com/s/19LAKD_E-e9vPufyBqTAZpQ 
提取码：4ddt 
## 数据集链接：https://pan.baidu.com/s/1UXuBBJ6Av8p_5eCcoLzdIA 
提取码：ocmo 

### 1  n01440764文件夹是原始数据集，约1300张包含人和鱼的图片，大小不同
### 2  generate_qualified_images.ipynb的目的是将n01440764文件中像素大于416x416的图片筛选出来，获取数量为200的合格样本存放         selected_images文件夹中。使用labelImg对两百张图片打标签，共有两类，分别是人脸和鱼的，生成xml文件。
### 3  check_annotations.ipynb的目的是检查selected_images文件夹中图片是否有漏标的xml文件，同时检查xml文件中是否有类别名称的拼写错  误。
### 4  compress_images.ipynb是将selected_images文件夹中图片统一压缩成416x416像素的图片，然后按照新图片扩张的比例修改xml文件中标记  的位置，并放入新的文件夹images_416x416
### 5  generateTxtFile.py是将images_416x416中的xml文本中的信息提取出来（images_416x416\191.jpg 98,180,334,314,0    169,110,224,187,1），并按照0.1的比例划分为训练和测试文档dataset_train.txt/dataset_test.txt
### 6  model_data文件夹是存储配置文件的，里面有目标个数文本和锚框文本。
   saved_model文件夹是放置训练权重文件夹。
   yolo3文件夹是已经搭建好的yolo v3的网络模型。
### 7  train.py是训练文件，训练好的权重文件trained_weights.h5保存在saved_model文件夹中
### 8  yolo.py定义了一个YOLO对象通过训练好的权重对图片中的目标检测，并将检测结果输出。
### 9  yolo_test.py测试文件，读入测试图片路径，调用对象方法，输出目标检测结果。
### 10  yolo_multiImages.py是连续对图片进行目标检测，并将输出结果合并成成一个mp4文件。
