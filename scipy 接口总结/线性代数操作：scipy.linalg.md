### 线性代数操作：[scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#scipy.linalg)

```python
from scipy import linalg
```

可以参考`numpy.linalg`:https://numpy.org/devdocs/reference/routines.linalg.html

对于`scipy.linalg`:

| 函数名称                                                     | 作用                                             |
| ------------------------------------------------------------ | ------------------------------------------------ |
| [`inv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv)（a [，overwrite_a，check_finite]） | 计算矩阵的逆                                     |
| [`solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve)（a，b [，sym_pos，lower，overwrite_a，...]） | 求解方阵的未知线性方程组                         |
| [`det`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.det.html#scipy.linalg.det)（a [，overwrite_a，check_finite]） | 计算矩阵的行列式                                 |
| [`norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html#scipy.linalg.norm)（a [，ord，axis，keepdims]） | 矩阵或矢量规范                                   |
| [`lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq)（a，b [，cond，overwrite_a，...]） | 计算方程Ax = b的最小二乘解                       |
| [`pinv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv.html#scipy.linalg.pinv)（a [，cond，rcond，return_rank，check_finite]） | 计算矩阵的（Moore-Penrose）伪逆                  |
| [`pinv2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinv2.html#scipy.linalg.pinv2)（a [，cond，rcond，return_rank，...]） | 计算矩阵的（Moore-Penrose）伪逆                  |
| [`pinvh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pinvh.html#scipy.linalg.pinvh)（a [，cond，rcond，lower，return_rank，...]） | 计算Hermitian矩阵的（Moore-Penrose）伪逆         |
| [`eig`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html#scipy.linalg.eig)（a [，b，left，right，overwrite_a，...]） | 求解方阵的普通或广义特征值问题。                 |
| [`eigvals`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvals.html#scipy.linalg.eigvals)（a [，b，overwrite_a，check_finite，...]） | 从普通或广义特征值问题计算特征值。               |
| [`lu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html#scipy.linalg.lu)（a [，permute_l，overwrite_a，check_finite]） | 计算矩阵的旋转LU分解。                           |
| [`svd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html#scipy.linalg.svd)（a [，full_matrices，compute_uv，...]） | 奇异值分解。                                     |
| [`svdvals`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svdvals.html#scipy.linalg.svdvals)（a [，overwrite_a，check_finite]） | 计算矩阵的奇异值。                               |
| [`ldl`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ldl.html#scipy.linalg.ldl)（A [，lower，hermitian，overwrite_a，...]） | 计算对称/埃尔米特矩阵的LDLt或Bunch-Kaufman分解。 |
| [`cholesky`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html#scipy.linalg.cholesky)（a [，lower，overwrite_a，check_finite]） | 计算矩阵的Cholesky分解。                         |
| [`qr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html#scipy.linalg.qr)（a [，overwrite_a，lwork，mode，pivoting，...]） | 计算矩阵的QR分解。                               |
| [`schur`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.schur.html#scipy.linalg.schur)（a [，输出，lwork，overwrite_a，sort，...]） | 计算Schur分解矩阵。                              |
| [`scipy.linalg.interpolative`](https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html#module-scipy.linalg.interpolative) | 插值矩阵分解                                     |

| 矩阵函数                                                     | 作用                                  |
| ------------------------------------------------------------ | ------------------------------------- |
| [`expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html#scipy.linalg.expm)（一个） | 使用Pade近似计算矩阵指数。            |
| [`logm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html#scipy.linalg.logm)（A [，disp]） | 计算矩阵对数。                        |
| [`cosm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cosm.html#scipy.linalg.cosm)（一个） | 计算矩阵余弦。                        |
| [`sinm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sinm.html#scipy.linalg.sinm)（一个） | 计算矩阵正弦。                        |
| [`tanm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.tanm.html#scipy.linalg.tanm)（一个） | 计算矩阵切线。                        |
| [`coshm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.coshm.html#scipy.linalg.coshm)（一个） | 计算双曲矩阵余弦。                    |
| [`sinhm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sinhm.html#scipy.linalg.sinhm)（一个） | 计算双曲矩阵正弦。                    |
| [`tanhm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.tanhm.html#scipy.linalg.tanhm)（一个） | 计算双曲矩阵切线。                    |
| [`signm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.signm.html#scipy.linalg.signm)（A [，disp]） | 矩阵标志功能。                        |
| [`sqrtm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html#scipy.linalg.sqrtm)（A [，disp，blocksize]） | 矩阵平方根。                          |
| [`funm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.funm.html#scipy.linalg.funm)（A，func [，disp]） | 评估可调用指定的矩阵函数。            |
| [`expm_frechet`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_frechet.html#scipy.linalg.expm_frechet)（A，E [，method，compute_expm，...]） | 在方向E上矩阵指数A的Frechet导数       |
| [`expm_cond`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_cond.html#scipy.linalg.expm_cond)（A [，check_finite]） | Frobenius范数中矩阵指数的相对条件数。 |
| [`fractional_matrix_power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fractional_matrix_power.html#scipy.linalg.fractional_matrix_power)（在） | 计算矩阵的分数幂。                    |

注：

- 线性方程组$aX+bY=Z$,求解$X，Y，Z$：

  `scipy.linalg.solve(a, b)`

  `a`表示$X$前系数，`b`表示$Y$前系数

  例如：

  ```python
  from scipy import linalg
  import numpy as np
  
  a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
  b = np.array([2, 4, -1])
  
  x = linalg.solve(a, b)
  print(x)
  ```

  输出：

  ```python
  [ 2. -2.  9.]
  ```

- 计算方阵$X$的行列式$|X|$:

  `scipy.linalg.det(x)`

  例如：

  ```python
  from scipy import linalg
  import numpy as np
  
  A = np.array([[1, 2],[3, 4]])
  x = lingalg.det(A)
  print(x)
  ```

- 计算特征值和特征向量：

  根据下述关系：
  $$
  Av=\lambda v
  $$
  `scipy.linalg.eig(A)`从普通或广义特征值问题计算特征值，返回特征值和特征向量。

  例如：

  ```python
  from scipy import linalg
  import numpy as np
  
  A = np.array([[1,2],[3,4]])
  
  l, v = linalg.eig(A)  #  l为特征值，v为特征向量
  print(l)
  print(v)
  ```

  输出：

  ```python
  [-0.37228132+0.j  5.37228132+0.j]
  [[-0.82456484 -0.41597356]
   [ 0.56576746 -0.90937671]]
  ```

- 奇异值分解：

  根据下述关系：
  $$
  M=U \Sigma V^*
  $$
  ​		其中$U$是$m\times m$阶酉矩阵，$\Sigma$是半正定$m \times n$阶对角矩阵，$V^*$是$V$的共轭转置（$n \times n$阶		酉矩阵）

  `scipy.linalg.svd()`将$M$分解为两个酉矩阵$U,V^*$以及奇异值的一维数组$\Sigma$。

  例如：

  ```python
  from scipy import linalg
  import numpy as np
  
  a = np.random.randn(3, 2) + 1.j*np.random.randn(3, 2)
  U, s, Vh = linalg.svd(a)
  print (U, Vh, s)
  ```

  输出：

  ```python
  [[-0.60142679+0.28212127j  0.35719830-0.03260559j  0.61548126-0.22632383j]
   [-0.00477296+0.44250532j  0.64058557+0.15734719j -0.40414313+0.45357092j]
   [ 0.46360086+0.38462177j -0.18611686+0.6337182j   0.44311251+0.06747886j]] [[ 0.98724353+0.j         -0.01113675+0.15882756j]
   [-0.15921753+0.j         -0.06905445+0.9848255j ]] [ 2.04228408  1.33798044]
  ```

  

