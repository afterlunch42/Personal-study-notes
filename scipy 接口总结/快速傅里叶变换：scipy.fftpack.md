### 快速傅里叶变换：[scipy.fftpack](https://docs.scipy.org/doc/scipy/reference/fftpack.html#scipy.fftpack)

快速傅里叶变换（FFT）是快速计算序列的离散傅里叶变换（DFT）或其逆变换的方法。

FFT会通过把DFT矩阵分解为稀疏因子之积来快速计算此类变换。

![](https://upload-images.jianshu.io/upload_images/147042-0071478b913118d7.gif?imageMogr2/auto-orient/strip%7CimageView2/2/w/300/format/webp)

**一维离散傅里叶变换**

长度为`N`的序列`x [n]`的`FFT y [k]`由`fft()`计算，逆变换使用`ifft()`计算

例如：

```python
import numpy as np
from scipy.fftpack import fft

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
y = fft(x)
print (y)
```

输出：

```python
[ 4.50000000+0.j          2.08155948-1.65109876j -1.83155948+1.60822041j
 -1.83155948-1.60822041j  2.08155948+1.65109876j]
```

```python
from scipy.fftpack import fft
from scipy.fftpack import ifft

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
y = fft(x)
yinv = ifft(y)
print (yinv)
```

输出：

```python
[ 1.0+0.j  2.0+0.j  1.0+0.j -1.0+0.j  1.5+0.j]
```

`scipy.fftpack`模块允许计算快速傅里叶变换。例如：

```python
import numpy as np
time_step = 0.02
period = 5.
time_vec = np.arange(0, 20, time_step)
sig = np.sin(2 * np.pi / period * time_vec) + 0.5*np.random.randn(time_vec.size)
print (sig.size)
```

我们以`0.02`秒的时间步长创建一个信号，最后一条语句显示信号`sig`的大小，输出结果如下：

```python
1000
```

我们不知道信号频率; 只知道信号`sig`的采样时间步长。 信号应该来自实际函数，所以傅里叶变换将是对称的。 `scipy.fftpack.fftfreq()`函数将生成采样频率，`scipy.fftpack.fft()`将计算快速傅里叶变换。

例如：

```python
from scipy import fftpack

sample_freq = fftpack.fftfreq(sig.size, d = time_step)
sig_fft = fftpack.fft(sig)
print (sig_fft)
```

输出：

```python
array([ 
   25.45122234 +0.00000000e+00j,   6.29800973 +2.20269471e+00j,
   11.52137858 -2.00515732e+01j,   1.08111300 +1.35488579e+01j,
   …….])
```

**离散余弦变换**

离散余弦变换(DCT)根据以不同频率振荡的余弦函数的和表示有限数据点序列。 SciPy提供了一个带有函数`idct`的DCT和一个带有函数`idct`的相应IDCT。

例如：

```python
from scipy.fftpack import dct
mydict = dct(np.array([4., 3., 5., 10., 5., 3.]))
print(mydict)
```

输出：

```python
[ 60.          -3.48476592 -13.85640646  11.3137085    6.          -6.31319305]
```

逆离散余弦变换从其离散余弦变换(DCT)系数重建序列。 `idct`函数是`dct`函数的反函数。

例如：

```python
from scipy.fftpack import dct
from scipy.fftpack import idct
d = idct(np.array([4., 3., 5., 10., 5., 3.]))
print(d)
```

输出：

```python
[ 39.15085889 -20.14213562  -6.45392043   7.13341236   8.14213562
  -3.83035081]
```

