# PCVch02

记录学习Python Computer Vision的过程

第二次

## SIFT原理

尺度不变特征转换(Scale-invariant feature transform或SIFT)是一种电脑视觉的[算法](http://lib.csdn.net/base/datastructure)用来侦测与描述影像中的局部性特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由 David Lowe在1999年所发表，2004年完善总结。

SIFT算法的实质是在不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向。SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如角点、边缘点、暗区的亮点及亮区的暗点等。

在运用SIFT算法时，首先要进行空间极值的检测，就是利用高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。再来，就是对这些关键点进行定位，主要通过一个拟合精细的模型来确定位置和尺度。其次是方向的确定，通过图像局部的梯度方向，分配给每个关键点位置一个或多个方向。最后就是对这些关键点进行描述，在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。

## SIFT特征

### 兴趣点检测

在进行兴趣点检测时候如用完整的python实现SIFT特征的所有步骤可能效率不高，所以这里使用了一个开源工具包VLFeat，关于VLFeat的安装在第一章已经有说过这里不再说明。

为了比较SIFT特征和Harris角点的不同用了如下代码来演示同一张图片

#### 代码

```py
# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from PCV.localdescriptors import sift
from PCV.localdescriptors import harris

# 添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

imname =r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch02\s01.jpg'
im = array(Image.open(imname).convert('L'))
sift.process_image(imname, 'empire.sift')
l1, d1 = sift.read_features_from_file('empire.sift')

figure()
gray()
subplot(131)
sift.plot_features(im, l1, circle=False)
title(u'SIFT特征',fontproperties=font)
subplot(132)
sift.plot_features(im, l1, circle=True)
title(u'用圆圈表示SIFT特征尺度',fontproperties=font)

# 检测harris角点
harrisim = harris.compute_harris_response(im)

subplot(133)
filtered_coords = harris.get_harris_points(harrisim, 6, 0.1)
imshow(im)
plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
axis('off')
title(u'Harris角点',fontproperties=font)

show()
```

#### 实验结果

![image](https://github.com/zengqq1997/PCVch02/blob/master/result.jpg)

我们可以看出，两个算法所选择的特征点的位置是不同的

### 匹配描述子

将一幅图像中的特征匹配到另一幅图像的特征，一种稳健的准则是使用这两个特征距离和两个最匹配特征距离的比率相比于图像中的其他特征，该准则保证能够找到足够相似的唯一特征。使用该方法可以使错误的匹配数降低。

下面代码实现了匹配函数，并且将Harris角点用来做比对

#### SIFT代码

```python
from PIL import Image
from pylab import *
import sys
from PCV.localdescriptors import sift


if len(sys.argv) >= 3:
  im1f, im2f = sys.argv[1], sys.argv[2]
else:
  im1f = r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch02\s01.jpg'
  im2f = r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch02\s02.jpg'
#  im1f = '../data/crans_1_small.jpg'
#  im2f = '../data/crans_2_small.jpg'
#  im1f = '../data/climbing_1_small.jpg'
#  im2f = '../data/climbing_2_small.jpg'
im1 = array(Image.open(im1f))
im2 = array(Image.open(im2f))

sift.process_image(im1f, 'out_sift_1.txt')
l1, d1 = sift.read_features_from_file('out_sift_1.txt')
figure()
gray()
subplot(121)
sift.plot_features(im1, l1, circle=False)

sift.process_image(im2f, 'out_sift_2.txt')
l2, d2 = sift.read_features_from_file('out_sift_2.txt')
subplot(122)
sift.plot_features(im2, l2, circle=False)

matches = sift.match(d1, d2)
matches = sift.match_twosided(d1, d2)
print '{} matches'.format(len(matches.nonzero()[0]))

figure()
gray()
sift.plot_matches(im1, im2, l1, l2, matches, show_below=True)
show()
```

#### Harris角点代码

```python
 # -*- coding: utf-8 -*-
from pylab import *
from PIL import Image

from PCV.localdescriptors import harris
from PCV.tools.imtools import imresize

"""
This is the Harris point matching example in Figure 2-2.
"""

# Figure 2-2上面的图
#im1 = array(Image.open("../data/crans_1_small.jpg").convert("L"))
#im2 = array(Image.open("../data/crans_2_small.jpg").convert("L"))

# Figure 2-2下面的图
im1 = array(Image.open(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch02\s01.jpg').convert("L"))
im2 = array(Image.open(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\ch02\s02.jpg').convert("L"))

# resize to make matching faster
im1 = imresize(im1, (im1.shape[1]/2, im1.shape[0]/2))
im2 = imresize(im2, (im2.shape[1]/2, im2.shape[0]/2))

wid = 5
harrisim = harris.compute_harris_response(im1, 5)
filtered_coords1 = harris.get_harris_points(harrisim, wid+1)
d1 = harris.get_descriptors(im1, filtered_coords1, wid)

harrisim = harris.compute_harris_response(im2, 5)
filtered_coords2 = harris.get_harris_points(harrisim, wid+1)
d2 = harris.get_descriptors(im2, filtered_coords2, wid)

print 'starting matching'
matches = harris.match_twosided(d1, d2)

figure()
gray() 
harris.plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
show()
```

#### 实验结果

SIFT

![images](https://github.com/zengqq1997/PCVch02/blob/master/siftresult02.jpg)

Harris

![images](https://github.com/zengqq1997/PCVch02/blob/master/Harrisresult.jpg)

通过检测和匹配特征点，我们可以将这些局部描述子应用到很多例子中。

### 可视化连接图像

我们首先通过是否具有匹配的局部描述子来定义图像间的连接，然后可视化这些连接情况。为了完成可视化，我们可以在图中显示这些图像，图的边代表连接。

为了能够完成这样的连接我们需要用到一个叫pydot的工具包。那么接下来介绍下如何安装pydot工具包

那么在安装pydot工具包前需要先装graphviz-2.28.0.msi的画图工具和pyparsing，而后再安装pydot

- 安装[graphviz-2.28.0.msi](https://download.csdn.net/download/yinxing408033943/4639569)
  1. 安装graphviz过程中， 选择for everyone 否则会出错
  2. 配置系统环境变量：C:\Program Files (x86)\Graphviz 2.28\bin添加到path中
- 安装[pyparsing-1.5.7.win32-py2.7.exe](https://pypi.org/project/pyparsing/1.5.7/#files)
- 安装[pydot](https://pypi.org/project/pydot2/1.0.33/#files)
  1. 解压
  2. cmd 到包所在位置，注意必须是 setup.py所在路径 
  3. 执行python setup.py install

这样就安装完成了

我们使用了十七张集美大学本部内的校园建筑图，总共有三组

代码如下

#### 代码

```python
# -*- coding: utf-8 -*-
from pylab import *
from PIL import Image
from PCV.localdescriptors import sift
from PCV.tools import imtools
import pydot

""" This is the example graph illustration of matching images from Figure 2-10.
To download the images, see ch2_download_panoramio.py."""

#download_path = "panoimages"  # set this to the path where you downloaded the panoramio images
#path = "/FULLPATH/panoimages/"  # path to save thumbnails (pydot needs the full system path)

download_path = r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\imgs'  # set this to the path where you downloaded the panoramio images
path = r'C:\Users\ZQQ\Desktop\advanced\study\computervision\images\imgs'  # path to save thumbnails (pydot needs the full system path)

# list of downloaded filenames
imlist = imtools.get_imlist(download_path)
nbr_images = len(imlist)

# extract features
featlist = [imname[:-3] + 'sift' for imname in imlist]
for i, imname in enumerate(imlist):
    sift.process_image(imname, featlist[i])

matchscores = zeros((nbr_images, nbr_images))

for i in range(nbr_images):
    for j in range(i, nbr_images):  # only compute upper triangle
        print 'comparing ', imlist[i], imlist[j]
        l1, d1 = sift.read_features_from_file(featlist[i])
        l2, d2 = sift.read_features_from_file(featlist[j])
        matches = sift.match_twosided(d1, d2)
        nbr_matches = sum(matches > 0)
        print 'number of matches = ', nbr_matches
        matchscores[i, j] = nbr_matches
print "The match scores is: \n", matchscores

# copy values
for i in range(nbr_images):
    for j in range(i + 1, nbr_images):  # no need to copy diagonal
        matchscores[j, i] = matchscores[i, j]

#可视化

threshold = 2  # min number of matches needed to create link

g = pydot.Dot(graph_type='graph')  # don't want the default directed graph

for i in range(nbr_images):
    for j in range(i + 1, nbr_images):
        if matchscores[i, j] > threshold:
            # first image in pair
            im = Image.open(imlist[i])
            im.thumbnail((100, 100))
            filename = path + str(i) + '.png'
            im.save(filename)  # need temporary files of the right size
            g.add_node(pydot.Node(str(i), fontcolor='transparent', shape='rectangle', image=filename))

            # second image in pair
            im = Image.open(imlist[j])
            im.thumbnail((100, 100))
            filename = path + str(j) + '.png'
            im.save(filename)  # need temporary files of the right size
            g.add_node(pydot.Node(str(j), fontcolor='transparent', shape='rectangle', image=filename))

            g.add_edge(pydot.Edge(str(i), str(j)))
g.write_png('whitehouse.png')

```

#### 实验结果

![image](https://github.com/zengqq1997/PCVch02/blob/master/pydotresult.jpg)

## 小结

此次实验主要包括，兴趣点检测，匹配描述子，可视化连接图像。在这些中使用了VLFeat，pydot等工具包
