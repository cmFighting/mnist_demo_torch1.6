# 基于Pytorch1.6的Mnist手写数字识别demo

## Code Structure
```cmd
--imgs
--models
main_window.py 图形化界面
show_dataset.py 展示部分mnist数据
test4dataset.py 测试模型在整个数据集上准确率
test4singleimg.py 测试单张图片的效果
train_based_torchvision.py 通过torchvision加载数据集来进行训练
```

## How to run
 * 在cmd中执行下列命令配置虚拟环境
```cmd
conda create -n torch1.6 python==3.6.10
conda activate torch1.6

conda install pytorch torchvision cudatoolkit=10.2 # GPU(可选)
conda install pytorch torchvision cpuonly
pip install opencv-python
pip install matplotlib
```
* 从训练到测试的顺序执行脚本即可

 `注意：`训练的时候将train_based_torchvision.py第10行中的data_path修改为自己电脑上的一个空文件夹
 
 
 ## TODO
 1. 添加图形化界面程序
 2. 添加从文件夹中读取数据集训练的代码
 3. 注释写详细一点