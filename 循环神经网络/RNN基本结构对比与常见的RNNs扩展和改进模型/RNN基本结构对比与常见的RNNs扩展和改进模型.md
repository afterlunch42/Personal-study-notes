## RNN基本结构对比与常见的RNNs扩展和改进模型

### 1. RNN基本结构

#### 1.1 经典RNN结构

单个序列如图所示：

<img width="200px" src="https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.2.jpg" alt="单个序列">

RNN为解决处理序列数据的问题，引入隐状态$h$ (hidden state) ，$h$可以对序列数据提取特征，接着再转换为输出。

计算$h_1$：

<img width="400px" src="https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.3.jpg">

​				注：圆圈代表向量，箭头表示对向量做变换。

RNN中，每步使用的参数$U,W,b$相同，如同$h_1$的计算过程，计算$h_2$:

<img width='400px' src="https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.4.jpg">

同理，计算$h_3,h_4$:

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.5.jpg'>

然后计算RNN的输出$y_1$，采用$softmax$作为激活函数，根据$y_n=f(Wx+b)$,得到$y_1$:

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.6.jpg'>

使用和$y_1$相同的参数$V, c$，得到$y_1, y_2, y_3, y_4$:

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.7.jpg'>

从以上结构可看出，RNN结构的输入和输出等长。



#### 1.2 vector-to-sequence结构

若处理的问题为**输入一个单独值，输出一个序列**,有两种建模方式：

**方式一**：可只在其中的某一个序列进行计算

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.9.jpg'>

**方式二**：把输入信息作为每个阶段的输入

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.10.jpg'>



#### 1.3 sequence-to-vector结构

若处理的问题为**输入一个序列，输出一个单独的值**，此时通常在最后的一个序列上进行输出变换：

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.8.jpg'>



#### 1.4 Encoder-Decoder结构

若处理的问题为**序列不等长**问题，建模步骤如下：

**步骤一**：

将输入数据编码成一个上下文向量$c$，这一部分称为Encoder，得到$c$最简单方式为把Encoder的最后一个隐状态赋值为$c$，还可以对最后的隐状态做一个变换得到$c$,也可以对所有的隐状态做变换。

<img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.12.jpg'>

**步骤二**:

用另一个RNN网络（通常称为Decoder）对其进行编码。

方法一是将步骤一中的$c$作为初始状态输入到Decoder：

<img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.13.jpg'>

方法二是将$c$作为Decoder的每一步输入：

<img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.14.jpg'>



#### 1.5 上述三种结构对比

| 网络结构 | 结构图示                                                     | 应用场景举例                                                 |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1 vs N   | <img width='300px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.9.jpg'> | 1. 从图像生成文字，输入为图像的特征，输出为一段句子。<br/>2. 根据图像生成语音或音乐，输入为图像特征，输出为一段语音或音乐 |
| N vs 1   | <img width='300px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.8.jpg'> | 1、输出一段文字，判断其所属类别<br/>2、输入一个句子，判断其情感倾向<br/>3、输入一段视频，判断其所属类别 |
| N vs M   | <img width='300px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.13.jpg'> | 1、机器翻译，输入一种语言文本序列，输出另外一种语言的文本序列<br/>2、文本摘要，输入文本序列，输出这段文本序列摘要<br/>3、阅读理解，输入文章，输出问题答案<br/>4、语音识别，输入语音序列信息，输出文字序列 |



#### 1.6 Attention机制

在上述通用的Encoder-Decoder结构中，Encoder把所有的输入序列都编码成一个统一的语义特征$c$再解码，因此，$c$中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。如机器翻译问题，当要翻译的句子较长时，一个$c$可能存不下那么多信息，就会造成翻译精度的下降。**Attention机制通过在每个时间输入不同的$c$来解决此问题。**

引入了Attention机制的Decoder中，有不同的$c$，每个$c$会自动选择与当前输出最匹配的上下文信息：

<img width='400px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.15.jpg'>

> 例如：
>
> 输入序列是“我爱中国”，要将此输入翻译成英文：
>
> 
>
> 假如用$a_{ij}$衡量Encoder中第$j$阶段的$h_j$和解码时第$i$阶段的相关性，$a_{ij}$从模型中学习得到，和Decoder的第$i-1$阶段的隐状态、Encoder 第$j$个阶段的隐状态有关，比如$a_{3j}$的计算示意如下所示：
>
> <img width='300px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.19.jpg'>
>
> 最终Decoder中第$i$阶段的输入的上下文信息 $c_i$来自于所有$h_j$对$a_{ij}$的加权和。
>
> <img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/6.16.jpg'>
>
> 在Encoder中，$h_1,h_2,h_3,h_4$分别代表“我”，“爱”，“中”，“国”所代表信息。翻译的过程中，$c_1$会选择和“我”最相关的上下午信息，如上图所示，会优先选择$a_{11}$，以此类推，$c_2$会优先选择相关性较大的$a_{22}$，$c_3$会优先选择相关性较大的$a_{33}，a_{34}$，这就是attention机制。



#### 1.7 标准RNN前向过程

以$x$表示输入，$h$是隐层单元，$o$是输出，$L$为损失函数，$y$为训练集标签。$t$表示$t$时刻的状态，$V,U,W$是权值，同一类型的连接权值相同。以下图为例进行说明标准RNN的前向传播算法：

<img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/rnnbp.png'>

对于$t$时刻： $$ h^{(t)}=\phi(Ux^{(t)}+Wh^{(t-1)}+b) $$ 其中$\phi()$为激活函数，一般会选择tanh函数，$b$为偏置。

$t$时刻的输出为： $$ o^{(t)}=Vh^{(t)}+c $$ 模型的预测输出为： $$ \widehat{y}^{(t)}=\sigma(o^{(t)}) $$ 其中$\sigma$为激活函数，通常RNN用于分类，故这里一般用$softmax$函数。



### 2. 常见的RNNs扩展和改进模型



#### 2.1 长短期记忆模型（long short-term memery, LSTM)

##### 2.1.1 LSTM产生原因

RNN在处理长期依赖（时间序列上距离较远的节点）时会遇到巨大的困难，因为计算距离较远的节点之间的联系时会涉及$Jacobian$矩阵的多次相乘，会造成梯度消失或者梯度膨胀的现象。为了解决该问题，研究人员提出了许多解决办法，例如ESN（Echo State Network），增加有漏单元（Leaky Units）等等。其中最成功应用最广泛的就是门限RNN（Gated RNN），而LSTM就是门限RNN中最著名的一种。有漏单元通过设计连接间的权重系数，从而允许RNN累积距离较远节点间的长期联系；而门限RNN则泛化了这样的思想，允许在不同时刻改变该系数，且允许网络忘记当前已经累积的信息。



##### 2.1.2  标准RNN与LSTM的区别

所有RNN都具有一种重复神经网络模块的链式的形式，在标准RNN中，重复模块如下（例如使用$tanh$层）：

<img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM1.png'>

LSTM同样具有重复神经网络模块链式形式，但重复模块形式不同：

<img width='500px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM2.png'>

注：图标含义：

<img width='450px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM3.png'>

上图中，每一条黑线传输着一整个向量，从一个节点的输出到其他节点的输入。粉色的圈代表 pointwise 的操作，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。合在一起的线表示向量的连接，分开的线表示内容被复制，然后分发到不同的位置。



##### 2.1.3 LSTM核心思想

LSTM的关键在于**细胞状态**，水平线在图上方贯穿运行。

<img width='800px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM4.png'>

LSTM 有通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM5.png)

LSTM拥有三个门，分别是**忘记层门**、**输入层门**和**输出层门**，来保护和控制细胞状态。

(1) **忘记层门**

作用对象：细胞状态 。

作用：将细胞状态中的信息选择性的遗忘。

操作步骤：该门会读取$h_{t-1}$和$x_t$，输出一个在 0 到 1 之间的数值给每个在细胞状态$C_{t-1}$中的数字。1 表示“完全保留”，0 表示“完全舍弃”。

![img](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM6.png)

(2) **输入层门**

作用对象：细胞状态。

作用：将新的信息选择性的记录到细胞状态中。

操作步骤：

​	步骤一：$sigmoid$层称“输入门层”，决定什么值我们将要更新。

​	步骤二：$tanh$层创建一个新的候选值向量$\tilde{C}_t$加入到状态中

<img width='600px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM7.png'>

​	步骤三：将$c_{t-1}$更新为$c_{t}$。将旧状态与$f_t$相乘，丢弃掉我们确定需要丢弃的信息。接着加上					$i_t * \tilde{C}_t$得到新的候选值，根据我们决定更新每个状态的程度进行变化。

<img width='600px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM8.png'>

(3) **输入层门**

作用对象：隐层$h_t$

作用：确定输出什么值

操作步骤：
​	步骤一：通过sigmoid 层来确定细胞状态的哪个部分将输出。

​	步骤二：把细胞状态通过 tanh 进行处理，并将它和 sigmoid 门的输出相乘，最终我们仅仅会输					出我们确定输出的那部分。

<img width='600px' src='https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM9.png'>



##### 2.1.4 LSTM流行的变体

**增加peephole连接**

在正常的LSTM结构中，Gers F A 等人提出增加peephole 连接，可以门层接受细胞状态的输入。

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM10.png)

**对忘记门和输入门进行同时确定**

不同于之前是分开确定什么忘记和需要添加什么新的信息，这里是一同做出决定。

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM11.png)

**Gated Recurrent Unit**

由Kyunghyun Cho等人提出的Gated Recurrent Unit (GRU)，其将忘记门和输入门合成了一个单一的更新门，同样还混合了细胞状态和隐藏状态，和其他一些改动。

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/LSTM12.png)



#### 2.2 Simple RNNs (SRNs)

1. SRNs是一个三层网络，其在隐藏层增加了上下文单元。下图中的y是隐藏层，u是上下文单元。上下文单元节点与隐藏层中节点的连接是固定的，并且权值也是固定的。上下文节点与隐藏层节点一一对应，并且值是确定的。
2. 在每一步中，使用标准的前向反馈进行传播，然后使用学习算法进行学习。上下文每一个节点保存其连接隐藏层节点上一步输出，即保存上文，并作用于当前步对应的隐藏层节点状态，即隐藏层的输入由输入层的输出与上一步的自身状态所决定。因此SRNs能够解决标准多层感知机(MLP)无法解决的对序列数据进行预测的问题。 SRNs网络结构如下图所示：

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/figure_6.6.1_1.png)

#### 2.3 Bidirectional RNNs

Bidirectional RNNs(双向网络)将两层RNNs叠加在一起，当前时刻输出(第t步的输出)不仅仅与之前序列有关，还与之后序列有关。例如：为了预测一个语句中的缺失词语，就需要该词汇的上下文信息。Bidirectional RNNs是一个相对较简单的RNNs，是由两个RNNs上下叠加在一起组成的。输出由前向RNNs和后向RNNs共同决定。如下图所示：

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/figure_6.6.2_1.png)

#### 2.4 Deep RNNs

Deep RNNs与Bidirectional RNNs相似，其也是又多层RNNs叠加，因此每一步的输入有了多层网络。该网络具有更强大的表达与学习能力，但是复杂性也随之提高，同时需要更多的训练数据。Deep RNNs的结构如下图所示： 

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/figure_6.6.3_1.png)



#### 2.5 Echo State Networks (ESNs)

##### 2.5.1 ESNs基本思想

1. 核心结构为一个随机生成、保持不变的储备池（Reservoir）。储备池是大规模随机生成稀疏连接(SD通常保持在1%~5%，SD表示储备池中互相连接的神经元占总神经元个数$N$的比例)的循环结构。
2. 从储备池到输出层的权值矩阵式唯一需要调整的部分。
3. 简单的线性回归便能够完成网络训练。

##### 2.5.2 ESNs基本思想

**基本思想**：使用大规模随机连接的循环网络取代经典网络的中间层，从而简化网络的训练过程。

**网络中的参数**：
（1）W-储蓄池中节点间连接权值矩阵

（2）Win-输入层到储备池之间连接权值矩阵，表明储备池中的神经元之间式相互连接的

（3）Wback-输出层到储备池之间的反馈连接权值矩阵，表明储备池会有输出层来的反馈

（4）Wout-输入层、储备池、输出层到输出层之间的连接权值矩阵，表明输出测光不仅与储备池连接，还与输入层和自己连接

（5）Woutbias-输出层的偏置项

ESNs结构：

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/figure_6.6.4_2.png)

#### 2.6 Gated Recurrent Unit Recurrent Neural Networks (GRUs)

GRUs是一般的RNNs的变型版本，其主要是从以下两个方面进行改进。

1. 以语句为例，序列中不同单词处的数据对当前隐藏层状态的影响不同，越前面的影响越小，即每个之前状态对当前的影响进行了距离加权，距离越远，权值越小。
2. 在产生误差error时，其可能是由之前某一个或者几个单词共同造成，所以应当对对应的单词weight进行更新。GRUs的结构如下图所示。GRUs首先根据当前输入单词向量word vector以及前一个隐藏层状态hidden state计算出update gate和reset gate。再根据reset gate、当前word vector以及前一个hidden state计算新的记忆单元内容(new memory content)。当reset gate为1的时候，new memory content忽略之前所有memory content，最终的memory是由之前的hidden state与new memory content一起决定。

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/figure_6.6.5_1.png)



#### 2.7 Clockwork RNNs(CW-RNNs)

CW-RNNs是RNNs的改良版本，其使用时钟频率来驱动。它将隐藏层分为几个块(组，Group/Module)，每一组按照自己规定的时钟频率对输入进行处理。为了降低RNNs的复杂度，CW-RNNs减少了参数数量，并且提高了网络性能，加速网络训练。CW-RNNs通过不同隐藏层模块在不同时钟频率下工作来解决长时依赖问题。将时钟时间进行离散化，不同的隐藏层组将在不同时刻进行工作。因此，所有的隐藏层组在每一步不会全部同时工作，这样便会加快网络的训练。并且，时钟周期小组的神经元不会连接到时钟周期大组的神经元，只允许周期大的神经元连接到周期小的(组与组之间的连接以及信息传递是有向的)。周期大的速度慢，周期小的速度快，因此是速度慢的神经元连速度快的神经元，反之则不成立。

​	CW-RNNs与SRNs网络结构类似，也包括输入层(Input)、隐藏层(Hidden)、输出层(Output)，它们之间存在前向连接，输入层到隐藏层连接，隐藏层到输出层连接。但是与SRN不同的是，隐藏层中的神经元会被划分为若干个组，设为$g$，每一组中的神经元个数相同，设为$k$，并为每一个组分配一个时钟周期$T_i\epsilon{T_1,T_2,...,T_g}$，每一组中的所有神经元都是全连接，但是组$j$到组$i$的循环连接则需要满足$T_j$大于$T_i$。如下图所示，将这些组按照时钟周期递增从左到右进行排序，即$T_1<T_2<...<T_g$，那么连接便是从右到左。例如：隐藏层共有256个节点，分为四组，周期分别是[1,2,4,8]，那么每个隐藏层组256/4=64个节点，第一组隐藏层与隐藏层的连接矩阵为64$\times$64的矩阵，第二层的矩阵则为64$\times$128矩阵，第三组为64$\times$(3$\times$64)=64$\times$192矩阵，第四组为64$\times$(4$\times$64)=64$\times$256矩阵。这就解释了上一段中速度慢的组连接到速度快的组，反之则不成立。

**CW-RNNs的网络结构如下图所示**：

![](https://github.com/scutan90/DeepLearning-500-questions/raw/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/img/ch6/figure_6.6.7_1.png)



参考文献：

[1]  [https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch06_循环神经网络(RNN)/第六章_循环神经网络(RNN).md#62-图解rnn基本结构](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch06_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN)/%E7%AC%AC%E5%85%AD%E7%AB%A0_%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(RNN).md#62-%E5%9B%BE%E8%A7%A3rnn%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%84)

[2]（美）伊恩·古德费洛著；赵申剑，黎彧君，符天凡，李凯译.深度学习[M].北京：人民邮电出版社.2017.