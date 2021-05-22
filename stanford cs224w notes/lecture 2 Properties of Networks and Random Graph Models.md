### lecture 2: Properties of Networks and Random Graph Models

#### 1. Network properties: How to measure a network

**Plan: Key Network Properties **

​		Degree distribution :$P(k)$

​		Path length: $h$

​		Clustering coefficient :$C$

​		Connected component :$s$



(1) Degree distribution $P(k)$:

probability that a randomly chosen node has degree $k$.
$$
N_k = \#nodes\ with\   degree\  k
$$
Normalized histogram(直方图)：
$$
P(k)=\frac{N_k}{N}
$$
![image-20200130085012119](../../AppData/Roaming/Typora/typora-user-images/image-20200130085012119.png)



(2) Paths in a Graph:

A path is a sequence of nodes in witch each node is linked to the next one.

$P_n= \{i_0,i_1，...,i_n \}$       $P_n=\{(i_0,i_1),(i_1,i_2),...,(i_{n-1},i_n)\}$

A path can intersect itself and pass through the same edge multiple times 



(3) Distance in a Graph

* Distance (shortest path, geodesic(大地线的(曲面上两点间距离最短的线)))

  between a pair of nodes is defined as the number of edges along the shortest path connecting the nodes 

  * if the two nodes are not connected ,the Distance is usually defined as infinite (or  zero )

    ![image-20200201110050922](../../AppData/Roaming/Typora/typora-user-images/image-20200201110050922.png)

* In directed graphs, paths need to follow the direction of the arrows(箭头)

  * consequence :

    Distance is not symmetric :$h_{B,C} \ne h_{C, B}$

![image-20200201110423228](../../AppData/Roaming/Typora/typora-user-images/image-20200201110423228.png)



(4) Network Diameter (直径)

* **Diameter**(直径)： The maximum (shortest path ) distance between any pair of nodes in a graph.

* **Average path length**(平均路径度) for a connected graph or a strongly  connected directed graph.
  $$
  \bar{h}=\frac{1}{2E_{max}}\sum_{i,j\ne i }h_{ij}
  $$
    		$h_{ij}$ is the Distance from node $i$ to node$j$

  ​		 $E_{max}$ is the max number of edges(total number of node pairs)=$\frac{n(n-1)}{2}$

  * Many times we compute the average only over the connected pairs of nodes (that is , we ignore "infinite" length paths)
  * Note that its measure also applied to (strongly) connected components of a graph

(5) Clustering Coefficient(聚类系数)

* **Clustering coefficient**(for undirected graphs):

  * How connected  are $i$'s neighbor to each other?

  * Node $i$ with degree $k_i$

    ​	Node $k_i(k_i-1)$is max number of edges between the $k_i $ neighbors.

  * $C_i=\frac{2e_i}{k_i(k_i-1) }$

    ​	where $e_i$is the number of edges between the neighbors of node $i$

      ![image-20200209140017062](../../AppData/Roaming/Typora/typora-user-images/image-20200209140017062.png)

  * $C_i \in [0， 1]$

    ​	clustering coefficient is undefined (or defined to be 0) for nodes with degree 0 or 1

* **Average clustering Coefficient **:
  $$
  C= \frac{1}{N}\sum_{i}^{N}C_i
  $$

(6)Connectivity（连通性）

* size of the largest connected component 
  * Largest set where any two vertices can be joined by a path 
* **Largest component =Giant(巨大的) component**

![image-20200209140342793](../../AppData/Roaming/Typora/typora-user-images/image-20200209140342793.png)

> Networks Examples:
>
> MSN Messager：
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209140625742.png" alt="image-20200209140625742" style="zoom:50%;" />
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209140730178.png" alt="image-20200209140730178" style="zoom: 50%;" />
>
> Degree Distribution：
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209153514360.png" alt="image-20200209153514360" style="zoom:50%;" />
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209153618420.png" alt="image-20200209153618420" style="zoom:50%;" />
>
> Clustering：
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209154024887.png" alt="image-20200209154024887" style="zoom: 67%;" />
>
> connected components：
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209154117580.png" alt="image-20200209154117580" style="zoom: 67%;" />
>
> Diameter of WCC：
>
> <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209154558185.png" alt="image-20200209154558185" style="zoom:50%;" />
>
> <center>Avg. path length 6.6 （90% of the nodes can be  reached in <8 hops）</center>
>result:
> 
><img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209155057875.png" alt="image-20200209155057875" style="zoom:50%;" />
> 
>Another example: PPI(protein to potein interaction ) Network
> 
>![image-20200209155246909](../../AppData/Roaming/Typora/typora-user-images/image-20200209155246909.png)
> 
>

#### 2. Models of Graphs

##### (1) Erdos-Renyi Random Graphs

The simplest Models of  Graphs .

**Two variants:**(变体)

* $G_{np}$: undirected graph on $n$ nodes where each edge $(u, v)$ appears i.i.d（独立同分布） with probability $p$
* $G_{nm}$: undirected graph with $n$ nodes and $m$ edges picked uniformly at random

a. **Random Graph Model**:

![image-20200209161910094](../../AppData/Roaming/Typora/typora-user-images/image-20200209161910094.png)

Properties of $G_{np}$:

​	Degree distribution :$P(k)$

​	Path length : $h$

​	Clustering Coefficient : $C$

And the value of the graph :

* Degree Distribution(分布):

  * Fact: Degree distribution of $G_{np} $ is binomial(二项分布的)

  * Let $P(k)$ denote the fraction of node with degree $k$:

    $$P(k)=C^{k}_{n-1}p^k(1-p)^{n-k-1}$$ (binomial)

    understanding:

    ![image-20200209162705470](../../AppData/Roaming/Typora/typora-user-images/image-20200209162705470.png)

  * Review(probability): Mean, variance of a binomial distribution:

    $E(x)=p(n-1)$

    $var(x)=\sigma ^2=p(1-p)(n-1)$

     近似计算（*by the law of large numbers*）：

    ![image-20200209163110631](../../AppData/Roaming/Typora/typora-user-images/image-20200209163110631.png)

* Clustering Coefficient:

  * Remember: $C_i= \frac{2e_{i}}{k_i(k_i-1)}$    (definition)

    ​	where $e_i$ is the number of edges between $i$'s neighbors.

  * Edges in $G_{np}$ appear i.i.d with prob.$p$

  * so ,expected $E(e_i) =p\frac{k_i(k_i-1)}{2}$

     $p$: each pair is connected with prob.$p$

    $ \frac {k_i(k_i-1)}{2}$:Number of distinct pairs of neighbors of node $i$ of degree $k_i$

  * Then :

$$
E[C_i]=\frac {pk_i(k_i-1)}{k_i(k_i-1)}=p=\frac{\bar{k}}{n-1} \approx \frac{\bar{k}}{n}
$$

​    ![image-20200209164423390](../../AppData/Roaming/Typora/typora-user-images/image-20200209164423390.png)

* Path length:
  $$
  O(\log n)
  $$
  

  Def :Expansion

  ![image-20200209164636821](../../AppData/Roaming/Typora/typora-user-images/image-20200209164636821.png)

  ![image-20200209164944391](../../AppData/Roaming/Typora/typora-user-images/image-20200209164944391.png)

  ![image-20200209165415676](../../AppData/Roaming/Typora/typora-user-images/image-20200209165415676.png)

  Erdos-Renyi avg. shortest path:

  ![image-20200209165541608](../../AppData/Roaming/Typora/typora-user-images/image-20200209165541608.png)

* Connectivity :

![image-20200209165733373](../../AppData/Roaming/Typora/typora-user-images/image-20200209165733373.png)

summery:

![image-20200209165845080](../../AppData/Roaming/Typora/typora-user-images/image-20200209165845080.png)

##### (2)The Small-World Model

Motivation: Can we have **high clustering** while also having **short paths**?

Clustering Implies Edge Locality:

![image-20200209170353347](../../AppData/Roaming/Typora/typora-user-images/image-20200209170353347.png)

The "Controversy"(争论):

![image-20200209170544408](../../AppData/Roaming/Typora/typora-user-images/image-20200209170544408.png)

such as :

![image-20200209170713665](../../AppData/Roaming/Typora/typora-user-images/image-20200209170713665.png)

Solution: The Small-World Model:

* Start with a **low-dimensional regular lattice**(格子)

  * in our case we are using a ring as a lattice :

    <img src="../../AppData/Roaming/Typora/typora-user-images/image-20200209171030060.png" alt="image-20200209171030060" style="zoom:50%;" />

  * Has high clustering coefficient

* Rewire(重新连线): Introduce randomness(''shortcut'')

  * Add/remove edges to create shortcut to join remote pare of the lattice
  * For each edge, with prob.$p$, move the other endpoint to a random node 

![image-20200209171355381](../../AppData/Roaming/Typora/typora-user-images/image-20200209171355381.png)

![image-20200209171436578](../../AppData/Roaming/Typora/typora-user-images/image-20200209171436578.png)

result:
![image-20200209171645833](../../AppData/Roaming/Typora/typora-user-images/image-20200209171645833.png)

Summary of the Small-World Model:

![image-20200209171839162](../../AppData/Roaming/Typora/typora-user-images/image-20200209171839162.png)



##### (3) Kronecker Graph Model

Generating large realistic graphs

* Idea: **Recursive Graph Generation**(递归图的世代)

  * How can we think of network structure recursively?

    Intuition: **Self-similarity** (自相似性)

    Objective is similar to part of itself: the whole has the same shape as one or more of the parts.

  * **Mimic recursive graph/ community growth:**

    ![image-20200209173118906](../../AppData/Roaming/Typora/typora-user-images/image-20200209173118906.png)

    **Attention**: Kronecker product is a way of generating self-similar matrices

    

*  Kronecker graphs：

  A recursive model of network structure

  ![image-20200209173609741](../../AppData/Roaming/Typora/typora-user-images/image-20200209173609741.png)

* Kronecker product： [百度百科：克罗内克积](https://baike.baidu.com/item/克罗内克积/6282573?fr=aladdin)

  * Definition:

    Kronecker product of matrices $A$ and $B$ is given by:

    ![image-20200209173852029](../../AppData/Roaming/Typora/typora-user-images/image-20200209173852029.png)

  * Define a Kronecker product of two graph as a Kronecker product of their adjacency matrices.

* **Kronecker graph** is obtained by growing sequence of graphs by iteration the Kronecker product over the initiator matrix $K_1$:

  ![image-20200209174323027](../../AppData/Roaming/Typora/typora-user-images/image-20200209174323027.png)

  where the $K_1 $:

  ![image-20200209174347038](../../AppData/Roaming/Typora/typora-user-images/image-20200209174347038.png)

  note:

  One can easily use multiple initiator matrices $(K_1',K_1'',K_1''' )$(even of different sizes)

* **Kronecker Initiator Matrices **:

  ![image-20200209174629370](../../AppData/Roaming/Typora/typora-user-images/image-20200209174629370.png)

  

* Stochastic(随机的) Kronecker Graphs：

  * Create $N_1\times N_1$ probability matrix $\Theta_1$

  * Compute the $k^{th}$ Kronecker power $\Theta _k$

  * For each entry (入口) $P_{uv}$ of $\Theta _k$ include an edge $(u, v)$ in $K_k$ with probability $P_{uv}$

    ![image-20200209175651129](../../AppData/Roaming/Typora/typora-user-images/image-20200209175651129.png)

  ![image-20200209175834540](../../AppData/Roaming/Typora/typora-user-images/image-20200209175834540.png)

  ![image-20200209175856943](../../AppData/Roaming/Typora/typora-user-images/image-20200209175856943.png)

* **Fast Kronecker generator algorithm **:

  * For generating *directed graphs*:

    * Insert 1 edge on graph $G$ on $n=2^m$ nodes

    * create normalized matrix $L_{uv}=\frac{\Theta_{uv}}{\sum_{op}\Theta_{op}}$

    * For $i=1,...,m$

      ​	start with $x=0$,$y=0$

      ​	pick a row/column $(u, v)$ with prob.$L_{uv}$

      ​	Descend into quadrant $(u, v)$ at level $i$ of $G$

      ​	(This means:$x+=u 2^{m-i},y+=v2^{m-i}$)

    * Add an edge $(x, y)$to $G$



Result:

![image-20200209180718636](../../AppData/Roaming/Typora/typora-user-images/image-20200209180718636.png)



