### 特殊函数：[scipy.special](https://docs.scipy.org/doc/scipy/reference/special.html#scipy.special)

```python
from scipy import special 
```

自带常用特殊函数

| 常用特殊函数                         |                                       |
| ------------------------------------ | ------------------------------------- |
| Airy functions                       | 艾里函数                              |
| Elliptic Functions and Integrals     | 椭圆函数和积分                        |
| Bessel Functions                     | 贝塞尔函数                            |
| Struve Functions                     | q司徒卢威函数                         |
| Raw Statistical Functions            | 原始统计函数（可以使用`scipy.stats`） |
| Information Theory Functions         | 信息论功能                            |
| Gamma and Related Functions          | 伽玛及相关函数                        |
| Error Function and Fresnel Integrals | 错误功能和菲涅耳积分                  |
| Legendre Functions                   | Legendre函数                          |
| Ellipsoidal Harmonics                | 椭圆谐波                              |
| Orthogonal polynomials               | 正交多项式                            |
| Hypergeometric Functions             | 超几何函数                            |
| Parabolic Cylinder Functions         | 抛物柱面函数                          |
| Mathieu and Related Functions        | Mathieu及相关函数                     |
| Spheroidal Wave Functions            | 球形波函数                            |
| Kelvin Functions                     | 开尔文函数                            |
| Combinatorics                        | 组合                                  |
| Lambert W and Related Functions      | Lambert W及相关功能                   |

- 贝塞尔函数，比如`scipy.special.jn()` (第n个整型顺序的贝塞尔函数)
- 椭圆函数 (`scipy.special.ellipj()` Jacobian椭圆函数, ...)
- Gamma 函数: scipy.special.gamma(), 也要注意 `scipy.special.gammaln()` 将给出更高准确数值的 Gamma的log。
- Erf, 高斯曲线的面积：scipy.special.erf()