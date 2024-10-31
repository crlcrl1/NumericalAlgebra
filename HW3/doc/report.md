# 数值代数第 3 次上机作业报告

> 陈润璘 2200010848

## 问题描述

在求解线性方程组和最小二乘问题时，QR 分解时一种常用的方法。在求解最小二乘问题时 $\min \|Ax-b\|_2$，虽然其解可以通过法方程 $A^TAx = A^Tb$ 求得，但是由于直接求解法方程的数值稳定性较差，因此我们通常会使用 QR 分解来求解。

在本次上机作业中，我们使用 QR 分解的方法来求解三个线性方程组，并与 Gauss 消去法或平方根法进行比较。同时，我们还会使用 QR 分解的方法来求解两个个最小二乘问题。

## 数值方法

我们可以将 $A$ 矩阵分解为

$$
A = Q \begin{pmatrix} R \\ 0 \end{pmatrix}
$$

其中 $Q$ 是正交矩阵，$R$ 是上三角矩阵。这样就可以将最小二乘问题改写为 $\min \|Rx - Q^Tb\|_2$，这个问题的解可以通过回代求得。

为了将 $A$ 矩阵分解为 $Q$ 和 $R$ 的乘积，我们可以使用 Householder 变换或者 Givens 变换，依次将 $A$ 的每一列的下三角部分变为 0。在本次作业中，由于涉及到的矩阵 $A$ 都是稠密矩阵，因此我们使用 Householder 变换来实现 QR 分解，以充分利用矩阵运算的并行性。

## 算法实现

在本次作业中，我们使用 Python 中的 numpy 库来实现 QR 分解，并求解线性方程组和最小二乘问题。

在 Householder 变换的实现中，为了防止出现数值精度损失，我们将 Householder 矩阵写为 $H = I - \beta v v^T$ 的形式，其中 $v$ 的第一个分量为 1，$\beta = 2 / \|v\|_2^2$。并且，在计算 $v = x_1 - \|x\|_2$ 时，为了防止两个大数相减导致下溢，我们将其改写为 $v = \frac{-(x_2^2+\dots+x_n^2)}{x_1+\|x\|_2}$。

具体的 Householder 变换和 QR 的代码见 `shared.py` 中的 `householder` 和 `qr` 函数。

## 数值结果

### 线性方程组

首先，同时使用 QR 分解和 Gauss 消去法求解 $Ax = b$，其中

$$
A = \begin{pmatrix}
6 & 1 & & & & \\
8 & 6 & 1 & & &\\
& 8 & 6 & 1 & &\\
& & \ddots & \ddots &\ddots  &\\
& & & 8 & 6 & 1\\
\end{pmatrix},\quad
b = \begin{pmatrix}
7\\
15\\
15\\
\vdots\\
15\\
14
\end{pmatrix}
$$

两种算法的解与精确解之差的 2-范数如下：

|     方法     |    2 维    |   12 维    |   24 维    | 48 维  |   84 维    |
|:----------:|:---------:|:---------:|:---------:|:-----:|:---------:|
|     QR     | 2.220e-16 | 2.405e-13 | 3.871e-09 | 0.065 |    nan    |
| 不选主元 Gauss | 2.220e-16 | 1.537e-13 | 6.296e-10 | 0.011 | 7.259e+08 |
| 列主元 Gauss  |    0.0    |    0.0    |    0.0    |  0.0  | 3.783e-06 |

然后，我们取
$$
A = \begin{pmatrix}
10 & 1 &&&&\\
1 & 10 & 1 &&&\\
& 1 & 10 & 1 &&\\
&& \ddots & \ddots & \ddots &\\
&&& 1 & 10 & 1\\
&&&& 1 & 10
\end{pmatrix},\quad
$$
是一个 $100\times 100$ 的正定矩阵 $b$ 是一个随机的向量，我们计算 $Ax - b$ 的 2-范数，结果如下：

|  方法  |   2-范数    |
|:----:|:---------:|
|  QR  |   6.933   |
| 平方根法 | 5.758e-14 |

最后，我们取 A 为一个 40 阶的 Hilbert 矩阵，比较 QR 分解和平方根法的结果：

|  方法  |  2-范数   |
|:----:|:-------:|
|  QR  | 357.841 |
| 平方根法 | 110.523 |

有上述结果可知，QR 分解的数值稳定性要优于不选主元的 Gauss 消去法，但是不如列主元的 Gauss 消去法。在求解正定矩阵的线性方程组时，QR 分解的数值稳定性要劣于平方根法。

### 最小二乘问题

首先，我们使用 QR 分解的方法求解一个二次多项式的拟合问题，原始数据如下：

| $t_i$ |  -1  | -0.75  | -0.5 |  0   |  0.25  | 0.5  |  0.75  |
|:-----:|:----:|:------:|:----:|:----:|:------:|:----:|:------:|
| $y_i$ | 1.00 | 0.8125 | 0.75 | 1.00 | 1.3125 | 1.75 | 2.3125 |

最终求得的拟合曲线如下为 $y = 1.0 + 1.0t + 1.0t^2$。

然后，我们使用 QR 分解求解一个实际的房产估价问题，得到的结果如下：

$$
y = 2.08 + 0.72 a_1 + 9.68 a_2 + 0.15 a_3 + 13.68 a_4 + 1.99 a_5 - 0.96 a_6 - 0.48 a_7 - 0.07 a_8 + 1.02 a_9 + 1.44 a_{10} + 2.90 a_{11}
$$