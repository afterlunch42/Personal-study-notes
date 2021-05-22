#### 文件输入/输出:[scipy.io](https://docs.scipy.org/doc/scipy/reference/io.html#scipy.io)

```python
from scipy import stats
```

（1）载入和保存matlab文件

```python
import numpy as np
from scipy import io as spio
a = np.ones((3, 3))
# 写入文件
spio.savemat('file.mat', {'a':a})  
# savemat函数将其被传入的第二个参数保存为以第一个参数为名称的mat文件中
# 读取文件
data = spio.loadmat('file.mat',struct_as_record=True)  
# loadmat函数读取文件并赋值为data
print(data['a'])
```

输出：

```python
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
```

（2）读取图片

```python
from scipy import misc
misc.imread('fname.png')
# Matplotlib也有类似的方法
import matplotlib.pyplot as plt
plt.imread('fname.png')
```

更多io操作：

- 加载文本文件：[numpy.loadtxt()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html#numpy.loadtxt)/[numpy.savetxt()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html#numpy.savetxt)
- 智能加载文本/csv文件：[numpy.genfromtxt()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt)/numpy.recfromcsv()
- 快速有效，但是针对numpy的二进制格式：[numpy.save()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html#numpy.save)/[numpy.load()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html#numpy.load)