### Lecture 3： Motifs and Structure Roles in Networks

![image-20200210151822408](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210151822408.png)

#### 1. Subgraphs

##### (1)subnetworks

Subnetworks , or subgraphs ,are the building blocks of networks

They have the power to characterize and discriminate(区别) networks. 

##### (2)Case Example of subgraphs 

![image-20200210094616922](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210094616922.png)

<center>isometric :同构的</center>
**Motifs:**

* For each subgraph:
  * image you have a metric(度量标准) capable of classifying the subgraph "significance"(more on that later )
    * negative values indicate(表明) **under-representation** 
    * positive values indicate **over-representation **

* we create a network significance profile(扼要描述):
  * A feature vector(特征向量) with values for all subgraph types.

* Next: compare profiles of different networks:
  * Regulatory network (gene regulation)
  * Neuronal network (synaptic connections)
  * World wide web (hyperlinks between pages)
  * Social network (friendships)
  * Language networks (word adjacency)

![image-20200210151536839](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210151536839.png)



#### 2. Network Motifs(动机)

* Network Motifs : "recurring(循环) ,significant pattern of interconnections"
* How to define a network motif:
  * **Pattern**: Small induced subgraph
  * **Recurring **:Found many times, i.e. with high frequency
  * **Significant **: More frequent than expected, i.e., in randomly generated networks (Erdos-Renyi random graphs, scale-free networks)

* Motifs:
  * Help us understand How networks work 
  * Help us predict operation and reaction of the network in a given situation 

> Examples:
>
> * **Feed-forward loops**: found in networks of neurons, where they neutralize "Biological noise"
>
>   ![image-20200210153214172](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210153214172.png)
>
> * **Parallel loops**: found in food webs
>
>   ![image-20200210153245076](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210153245076.png)
>
> * **Single-input modules**: found in gene control networks
>
>   ![image-20200210153252936](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210153252936.png)

![image-20200210154441308](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210154441308.png)

* **Induced subgraph **:

  Induced subgraph of graph G is a graph, formed from a subset X of the vertices of graph G and all of the edges connecting pairs of vertices in subset X

* **Recurrence **:

  ![image-20200210154915483](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210154915483.png)

* Significance of a Motif:

  * key idea:

  Subgraphs than occur in a real network much more often than in a random network have functional significance 

  ![image-20200210155221844](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210155221844.png)

  * Motifs are overrepresented in a network when compared to randomized networks:

    
$$
Z_i=\frac{N_i^{real}-\bar{N}_i^{rand}}{std(N_i^{rand})}
$$

![image-20200210155507308](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210155507308.png)

  * **Network significance profile(SP)**:

$$
SP_i=\frac{Z_i}{\sqrt{\sum_{j} Z_j ^2}}
$$

![image-20200210160146386](C:\Users\67323\AppData\Roaming\Typora\typora-user-images\image-20200210160146386.png)





* **Configuration Model**:
  * **Goal**: Generate(生成) a random graph with a given degree sequence $k_1， k_2，...,k_N$
  * Useful as a "null" Model of networks:
    * we can compare the real network $G^{real}$and a "random" $G^{rand}$ which has the same degree sequence as $G^{real}$
  *  Configuration model:

![image-20200212110049406](../../AppData/Roaming/Typora/typora-user-images/image-20200212110049406.png)

<center>we ignore double edges and self-loops when creating the final graph</center>
​			random: Every node has the chance to connect other nodes whoever the node is .

* Alternative(交替的) for Spokes: **Switching** 

  * Start from a given graph $G$

  * Repeat the switching step $Q\times |E|$ times :

    * Select a pair of edges $A \to B$,$C \to D$ at random 

    * Exchange the endpoint to given $A \to D,C\to B$

      Exchange edges only if no multiple edges or self-edges are generated

      ![image-20200212111420762](../../AppData/Roaming/Typora/typora-user-images/image-20200212111420762.png)

  * **Result**: A randomly rewired graph:

    * Same node degrees, randomly rewired edges

  * $Q$ is chosen large enough (e.g., $Q=100$) for the process to converge(聚集，收敛)

* RECAP: Detecting Motifs:

  * Count subgraphs $i$ in $G^{real}$

  * Count subgraphs $i$ in random network $G^{rand}$

    * Configuration model: Each $G^{rand} $has the same #{nodes}, #{edges} and #{degree distribution} as $G^{real}$

  * Assign Z-score to $i$:

    * $Z_i=\frac{N_i^{real}-\bar{N_i}^{rand}}{std(N_i^{rand})}$

    * High Z-score : Subgraph $i$ is a network motif of $G$

      ![image-20200212113436725](../../AppData/Roaming/Typora/typora-user-images/image-20200212113436725.png)



* Variations on the Motif Concept:

  * Canonical definition

    * Directed and undirected 

    * Colored and uncolored 

    * Temporal and static motifs

      ![image-20200212113728750](../../AppData/Roaming/Typora/typora-user-images/image-20200212113728750.png)

  * Variations on the concept

    * Different frequency concepts

    * Different significance metrics

    * Under-Representation(anti-motifs)

    * Different constraints for null model

      ![image-20200212113918993](../../AppData/Roaming/Typora/typora-user-images/image-20200212113918993.png)

      

#### 3. Graphlets: Node feature vectors

* **Graphlets**: connected non-isomorphic subgraphs

  * Induced subgraphs of any frequency

    ![image-20200212115330336](../../AppData/Roaming/Typora/typora-user-images/image-20200212115330336.png)

  * Next: Use graphlets to obtain a node-level subgraph metric

  * Degree count #(edge) that a node touches:

    * we can generalize this noting for graphlets

  * Graphlet degree vector count #(graphlets) that a node touches

* Automorphism Orbit

  * An automorphism(自守) orbit takes into account the symmetries(对称性) of a subgraph
  * Graphlet Degree Vector（GVD）: a vector with the frequency of the node in each orbit position
  * Example: Graphlet degree vector of node $v$

![image-20200214145037590](../../AppData/Roaming/Typora/typora-user-images/image-20200214145037590.png)

* Graphlet Degree Vector (GDV)
  * Graphlet degree vector count #(graphlets) that a node touches at a particular orbit 
  * Considering Graphlet on 2 to 5 nodes we get 
    * Vector of 73 coordinates is  a signature of a node that describes the topology of node's neighborhood 
    * Captures its interconnectivities out to a distance of 4 hops。
  * Graphlet degree vector provides a measure of a node's local network topology:
    * Comparing vectors of two nodes provides a highly constraining measure of local topological similarity(局部拓扑相似性) between them .

![image-20200214150138987](../../AppData/Roaming/Typora/typora-user-images/image-20200214150138987.png)

#### 4. Finding Motifs and Graphlets

* Finding Motifs and Graphlets
  * Finding size-k motifs/graphlets requires solving two challenges:
    * 1) **Enumerating**(枚举) all size-k connected subgraphs 
    * 2) **Counting** #(occurrences of each subgraph type)
  * Just knowing if a certain subgraph exists in a graph is a hard computational problem
    * subgraph isomorphism is NP-complete
  * Computation time grows exponentially(指数的) as the size of the motif/Graphlet increases
    * feasible motifs size is usually small (3 to 8)
* counting subgraphs:
  * Network-centric approaches:
    * 1) **Enumerating**(枚举) all size-k connected subgraphs 
    * 2) **Counting** #(occurrences of each subgraph type) via graph isomorphisms test
  * Algorithms:
    * Exact subgraph enumeration (ESU) [Wernicke 2006] (精确子图枚举算法)
    * Kavosh [Kashani et al. 2009]
    * Subgraph sampling [Kashtan et al. 2004]
  * Today: **ESU algorithm**

> **Exact Subgraph Enumeration (ESU)**: (精确子图枚举算法)
>
> * **Two sets** :
>
>   * $V_{Subgraph}$: currently constructed subgraph (motif)
>   * $V_{extension}$: set of candidate(参考) nodes to extend the motif
>
> * **Idea**：
>
>   Starting with a node $v$, add those nodes $u$ to $V_{extension}$ set that nave two properties:
>
>   + $u$'s node_id must be larger than that of $v$
>   + $u$ may only be neighbored to some newly added node $w$ but not of any node in $V_{subgraph}$（extend ）
>
> * ESU is implemented(实施) as  a **recursive**(递归) **function**:
>
>   * The running of this function can be displayed as a tree-like structure of depth $k$ called the **ESU-Tree**
>
> ![image-20200214152516704](../../AppData/Roaming/Typora/typora-user-images/image-20200214152516704.png)
>
> Example:
>
> ![image-20200214152732746](../../AppData/Roaming/Typora/typora-user-images/image-20200214152732746.png)
>
> Use ESU-Tree to Count Subgraphs:
>
> ![image-20200214153137267](../../AppData/Roaming/Typora/typora-user-images/image-20200214153137267.png)
>
> ​	Classify subgraphs placed in the ESU-Tree leaves into non-isomorphic size-k classes:
>
> + Determine which subgraphs in ESU-Tree leaves are topologically equivalent(等价)(isomorphic) and group them into subgraph classes accordingly.
> + Use McKay's nauty algorithm [McKay 1981]

* Graph Isomorphsim(图同构)

  * Graphs $G$ and $H$ are isomorphic if there exists a bijection(双射) $f:V(G)\to V(H)$such that：
    * Any two nodes $u$ and $v$ of $G$ are adjacent in $G$ if $f(u)$ and $f(v)$ are adjacent in $H$
  * Example : Are $G$ and $H$ topologically equivalent?

  ![image-20200214154239045](../../AppData/Roaming/Typora/typora-user-images/image-20200214154239045.png)

#### 5. Structure Roles in Networks :

* Roles are "functions" of nodes in a network. Roles are measured by structural behaviors 

  ![image-20200214161244064](../../AppData/Roaming/Typora/typora-user-images/image-20200214161244064.png)

* Roles versus Groups in Networks :

  * Role : A collection of nodes which have similar positions in a network:

    * Roles are based on the similarity of ties between subset of nodes 

    * Different from Groups/communities 

      Communities/ Groups : A group of nodes that are well-connected to each other 

      * Group is formed based on adjacency , proximity or reachability
      * This is typically adopted in current data mining

  Nodes with the same role need not be in direct or even indirect interaction with each other.

* Structural Equivalence：

![image-20200214161911848](../../AppData/Roaming/Typora/typora-user-images/image-20200214161911848.png)

example：

![image-20200214162057122](../../AppData/Roaming/Typora/typora-user-images/image-20200214162057122.png)

* Why are Roles important?

![image-20200214162203440](../../AppData/Roaming/Typora/typora-user-images/image-20200214162203440.png)

* **Structural Role Discovery Method**

> * **RolX**:
>
>   Automatic discovery of nodes' structural roles in networks[Henderson, et al. 2011b]
>
>   + Unsupervised learning approach
>   + No prior knowledge required
>   + Assigns a mixed-membership of Roles to each node 
>   + Scales linearly in #(edge)
>
>   ![image-20200214162608619](../../AppData/Roaming/Typora/typora-user-images/image-20200214162608619.png)
>
>   Approach Overview:
>
>   ![image-20200214162732040](../../AppData/Roaming/Typora/typora-user-images/image-20200214162732040.png)
>
>   
>
> * **Recursive Feature Extraction**:
>
>   Recursive Feature Extraction [Henderson et al. 2011a] turns network connectivity into structure features
>
>   ![image-20200214162939027](../../AppData/Roaming/Typora/typora-user-images/image-20200214162939027.png)
>
>   * idea : Aggregate feature of a node and use them to generate new recursive features
>
>   * Base set of a node's neighborhood feature:
>
>     * **Local features **: all  measures of the node degree:
>
>       * if  network is directed, include in- and out-degree , total degree 
>       * if network is weighted , include weighted feature versions 
>
>     * **Egonet features** : Computed on the node's egonet:
>
>       * Egonet includes the node, its neighbors, and any edges in the included subgraph on these nodes 
>
>       * #(within-Egonet edges)
>
>         #(edges entering/leaving egonet) 
>
>   * algorithm :
>
> ![image-20200214163555102](../../AppData/Roaming/Typora/typora-user-images/image-20200214163555102.png)
>
> * Role Extraction :
>
> ![image-20200214163828664](../../AppData/Roaming/Typora/typora-user-images/image-20200214163828664.png)

![image-20200214163734669](../../AppData/Roaming/Typora/typora-user-images/image-20200214163734669.png)

Structural  sim: Co-authorship Net:

![image-20200214163951381](../../AppData/Roaming/Typora/typora-user-images/image-20200214163951381.png)