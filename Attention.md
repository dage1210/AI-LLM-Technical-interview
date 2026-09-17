
## Transformer 中的自注意力机制（Self-Attention）通过计算序列内部任意两个位置之间的关联度，使模型能够忽略词与词之间的距离，直接捕获全局语义依赖。

---

### 一、自注意力机制的工作原理

自注意力机制的计算过程可以划分为 4 个核心步骤：

<img width="434" height="708" alt="image" src="https://github.com/user-attachments/assets/61476708-fa84-474c-89e4-74e320e902ff" />


1. **线性投影生成 Q、K、V 向量**
对于输入序列的每一个词向量 $X$，分别乘以三个可学习的权重矩阵 $W^Q, W^K, W^V$，映射生成三个向量：
* **Query（查询向量 $Q$）：** 代表当前词“想要寻找什么信息”。
* **Key（键向量 $K$）：** 代表当前词“能提供什么特征”。
* **Value（值向量 $V$）：** 代表当前词“实际包含的表示内容”。


2. **计算注意力打分（Scaled Dot-Product）**
用 $Q$ 与所有位置的 $K$ 进行点积，衡量当前词与序列中其他词的相关性。为了防止向量维度 $d_k$ 较大时点积数值过大导致 Softmax 梯度消失，除以 $\sqrt{d_k}$ 进行缩放：

$\text{Score} = \frac{Q K^T}{\sqrt{d_k}}$


3. **Softmax 归一化**
对得分按行应用 Softmax 函数，转化为概率分布（权重之和为 1），得到注意力权重矩阵 $\text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$。
4. **加权求和输出**
将注意力权重乘以对应的 $V$ 向量并累加，生成融合了上下文信息的新特征表示：

$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$



---

### 二、为什么比 RNN 更适合处理长序列？

对比传统循环神经网络（RNN/LSTM），Self-Attention 在处理长序列时具有三大核心优势：

| 维度 | 自注意力机制（Self-Attention） | 循环神经网络（RNN） |
| --- | --- | --- |
| **路径长度 (Path Length)** | **$O(1)$**：任意两词间信息传递路径均为 1，有效解决长距离依赖衰减和梯度消失。 | **$O(N)$**：信息需按时序逐步传递，长距离信息易被稀释或遗忘。 |
| **并行计算能力** | **矩阵并行计算**：所有 Token 的 $Q, K, V$ 和注意力计算可一步完成，高度适配 GPU 硬件。 | **串行依赖**：第 $t$ 步的隐藏状态必须等待第 $t-1$ 步计算完成，无法充分利用并行算力。 |
| **计算复杂度 (每层)** | $O(N^2 \cdot d)$：与序列长度 $N$ 的平方成正比。 | $O(N \cdot d^2)$：与序列长度 $N$ 呈线性关系。 |

**总结：**

RNN 受限于**时序串行依赖**与**长距离信息衰减**，在序列增长时训练极其缓慢且容易遗忘早期信息；而 Self-Attention 通过直接连接任意距离的 Token 并利用**全并行矩阵运算**，极大地提升了长文本语义建模的能力与训练效率。

## 传统循环神经网络（RNN/LSTM），Self-Attention 在处理长序列时具有三大核心优势

在处理长序列时，Self-Attention（自注意力机制）相比传统 RNN/LSTM 展现出决定性的优势，主要体现在**计算路径距离**、**计算架构（并行性）**以及**信息表示与梯度流**三个维度。

---

### 1. 交互路径距离：
 $O(1)$ 
 vs 
 $O(N)$

长序列建模的核心难题在于如何让距离较远的信息发生交互。

```
RNN (串行传递，路径 = N):
[x1] ──> [h1] ──> [h2] ──> ... ──> [hN-1] ──> [hN]  (必须走 N 步)

Self-Attention (直接交互，路径 = 1):
[x1] ─────────────────────────────────────────> [xN]  (只需 1 步)

```

* **RNN / LSTM：** 信息需要沿着序列按时间步逐个传递。如果要让序列第 1 个位置的信息影响第 $N$ 个位置，信息必须经过 $N-1$ 个中间隐藏状态 $h_1 \to h_2 \to \dots \to h_N$ 。最大信息传递路径长度为 **$O(N)$**。随着 $N$ 的增大，早期信息在经过多次非线性变换和矩阵乘法后会严重衰减或失真（即“信息瓶颈”）。
* **Self-Attention：** 任意两个位置的 Token 之间通过 $Q$ 和 $K$ 的点积直接建立计算连接，**信息一步直达**。最大信息传递路径长度为 **$O(1)$**。这意味着无论序列长度是 100 还是 10,000，第 1 个词和最后一个词交互的“距离”完全相同，极大地缓解了长距离依赖衰减的问题。

---

### 2. 计算架构与硬件适配：打破时序串行锁

* **RNN / LSTM（严格串行）：**
计算当前时间步 $h_t$ 依赖于上一步的输出 $h_{t-1}$：

 $h_t = f(W_h h_{t-1} + W_x x_t + b)$ 



这种**时间上的前向依赖**使得 RNN 无法在序列维度上并行计算。GPU 拥有数千个并行核心，但在处理长序列 RNN 时，绝大多数算力处于等待状态，训练时间随序列长度 $N$ 呈线性增长，无法规模化。
* **Self-Attention（全并行计算）：**
将整个序列打包为输入矩阵 $X \in \mathbb{R}^{N \times d}$，一次性通过矩阵乘法并行生成所有位置的 $Q, K, V$：

$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$



注意力权重矩阵的计算 $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ 属于大规模 Dense 矩阵运算，能够完全吃满 GPU / TPU 的张量计算单元（Tensor Cores），使训练速度获得数量级提升。

---

### 3. 梯度传播与记忆机制：克服梯度消失/爆炸

* **RNN / LSTM 的梯度衰减：**
在反向传播（BPTT）过程中，梯度沿着时间轴反向穿透 $N$ 个时间步。以简单 RNN 为例，
$\frac{\partial h_N}{\partial h_1}$
涉及多次权重矩阵 $W_{hh}^T$ 的连乘：

$\frac{\partial h_N}{\partial h_1} = \prod_{t=2}^{N} \frac{\partial h_t}{\partial h_{t-1}}$



如果特征值小于 1，梯度指数级衰减至 0（梯度消失）；如果大于 1，则梯度爆炸。LSTM 和 GRU 引入门控机制（如 Cell State 的加法更新 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$）极大地改善了这一问题，但本质上依然属于**串行马尔可夫链**结构，面对数千级别的超长序列时依然难以保持极长期的梯度流动。
* **Self-Attention 的梯度捷径：**
由于输出 $Y_i = \sum_{j} A_{ij} V_j$，任何一个位置的梯度可以直接沿着注意力权重 $A_{ij}$ **直接回传**给任意位置的输入向量，不需要经过中间时间步的连乘。这种全连接的结构天然为反向传播提供了“高速公路”。

---

### 总结对比与使用权衡

| 对比维度 | 传统 RNN / LSTM | Self-Attention (Transformer) |
| --- | --- | --- |
| **最大信息路径** | $O(N)$（随序列长度增加） | $O(1)$（恒定） |
| **计算时间复杂度** | $O(N \cdot d^2)$ | $O(N^2 \cdot d)$ |
| **计算并行性** | 无法在序列维度并行 | 序列维度全并行 |
| **长距离依赖捕获** | 极难，受限于容量与时间瓶颈 | 极强，全局直接感知 |
| **瓶颈与短板** | 训练慢、易遗忘早期上下文 | 显存与计算量随长度呈平方级（ $N^2$ ）增长 |

虽然 Self-Attention 在长序列处理上优势明显，但其**平方级别的计算复杂度 $O(N^2)$** 也带来了显存和计算量的挑战。这也是为何后续会出现 FlashAttention、线性注意力（Linear Attention）以及 Sparse Attention 等优化手段，旨在保留其 $O(1)$ 路径优势的同时降低计算开销。


## 计算时间复杂度的计算

Transformer 中的 Self-Attention 和传统 RNN 的**单层时间复杂度**推导过程如下：

---

### 一、Self-Attention 的复杂度：
$O(N^2 \cdot d)$

假设序列长度为 $N$，特征维度（词向量维度）为 $d$。Self-Attention 的计算可以拆解为以下三个核心步骤：

#### 步骤 1：生成 Q、K、V 矩阵

* **计算过程：** 输入矩阵 $X \in \mathbb{R}^{N \times d}$ 分别乘以三个权重矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$。
* **计算量：**
一个 $(N \times d)$ 矩阵与 $(d \times d)$ 矩阵相乘，单次矩阵乘法需要进行 $N \times d \times d = N \cdot d^2$ 次乘加运算。
* **小计：** 计算 $Q, K, V$ 三个矩阵的总计算量为：

$3 \times (N \cdot d^2) = O(N \cdot d^2)$



#### 步骤 2：计算注意力打分矩阵 
 $Q \cdot K^T$

* **计算过程：** $Q \in \mathbb{R}^{N \times d}$ 与 $K^T \in \mathbb{R}^{d \times N}$ 相乘，得到注意力得分矩阵 $A \in \mathbb{R}^{N \times N}$。
* **计算量：**
$(N \times d)$ 矩阵与 $(d \times N)$ 矩阵相乘，需要的计算量为：

$N \times d \times N = N^2 \cdot d$


* **Softmax 归一化：** 对 $N \times N$ 矩阵按行计算 Softmax，开销为 $O(N^2)$，相比矩阵乘法可忽略。
* **小计：** 这一步的计算量为 $O(N^2 \cdot d)$。

#### 步骤 3：加权求和得到输出 
$A \cdot V$

* **计算过程：** 注意力权重矩阵 $A \in \mathbb{R}^{N \times N}$ 与 $V \in \mathbb{R}^{N \times d}$ 相乘，得到最终输出矩阵 $O \in \mathbb{R}^{N \times d}$。
* **计算量：**
$(N \times N)$ 矩阵与 $(N \times d)$ 矩阵相乘，需要的计算量为：

$N \times N \times d = N^2 \cdot d$


* **小计：** 这一步的计算量为 $O(N^2 \cdot d)$。

#### 汇总

将三步相加，总计算量为：


$\text{Total} = O(N \cdot d^2) + O(N^2 \cdot d) + O(N^2 \cdot d) = O(N \cdot d^2 + N^2 \cdot d)$

在实际应用（如 LLM、大语言模型）中，**序列长度 $N$ 通常远大于维度 $d$**（例如 $N = 4096, d = 512$ 或 $N = 32000, d = 4096$）。此时主导项为平方项，因此整体计算复杂度记作：


$\mathbf{O(N^2 \cdot d)}$

---

### 二、传统 RNN 的复杂度：
$O(N \cdot d^2)$

假设序列长度为 $N$，隐藏层特征维度为 $d$（为简化推导，假设输入维度与隐藏层维度一致均等于 $d$）。

RNN 在每个时间步 $t$ 需要计算：


$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$

#### 单个时间步 $t$ 的计算量：

1. **隐藏状态变换 $W_{hh} h_{t-1}$：**
权重矩阵 $W_{hh} \in \mathbb{R}^{d \times d}$ 与前一时刻隐藏向量 $h_{t-1} \in \mathbb{R}^{d \times 1}$ 相乘，计算量为 $d \times d = d^2$。
2. **输入特征变换 $W_{xh} x_t$：**
权重矩阵 $W_{xh} \in \mathbb{R}^{d \times d}$ 与当前输入向量 $x_t \in \mathbb{R}^{d \times 1}$ 相乘，计算量为 $d \times d = d^2$。
3. **向量相加与激活函数：** 复杂度为 $O(d)$，相比  $d^2$  项可忽略。

* **单时间步小计：**
  $d^2 + d^2 = 2d^2 = O(d^2)$ 。

#### 整个序列（共 $N$ 个时间步）：

由于 RNN 必须按时间步逐个串行计算，总共需要重复执行 $N$ 次：


$\text{Total} = N \times O(d^2) = \mathbf{O(N \cdot d^2)}$

注：如果是 LSTM，内部包含 4 个门控结构，计算量为  
$4 \times N \cdot d^2$ 
，但大 O 表示法下常数系数省略，依然为  
$O(N \cdot d^2)$ 
。

---

### 三、两者复杂度的核心对立

通过上述计算过程可以看出两者瓶颈的不同：

* **RNN 瓶颈在维度  $d$ ：** 计算量与序列长度  $N$  呈**线性关系**，但由于串行依赖无法利用 GPU 的全并行能力。
* **Self-Attention 瓶颈在长度 $N$：** 矩阵乘法虽然能极高效率地并行计算，但由于计算
   $Q \cdot K^T$
  产生了
  $N \times N$
  的注意力图，导致计算量与显存开销随序列长度
  $N$
  呈**平方级爆炸** 
  （ $N^2$ ）
  。

## 多头不会增加计算量
**结论：**

**计算复杂度没有发生变化，依然是 $O(N^2 \cdot d + N \cdot d^2)$。**

分化成 $h$ 个头（Multi-Head）只是将维度 $d$ 拆分到了不同的子空间中，**总的计算量（FLOPs）在数学上完全一致**。

---

### 推导过程

假设：

* 序列长度为 $N$
* 模型隐藏层维度（特征总维度）为 $d$
* 头数为 $h$
* 每个头的特征维度为 $d_k = \frac{d}{h}$

我们可以按照多头注意力机制（Multi-Head Attention）的计算步骤一步步推导：

#### 步骤 1：生成所有头的 Q、K、V 矩阵

在实际实现中，通常**不会**显式开辟 $h$ 个独立的矩阵进行 $h$ 次小矩阵乘法，而是**先通过一次大矩阵乘法生成总的 Q、K、V，再进行形变（Reshape / Split）**。

* **计算过程：**
输入 $X \in \mathbb{R}^{N \times d}$ 乘以权重矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$，得到 $Q, K, V \in \mathbb{R}^{N \times d}$。
随后将 $Q, K, V$ 重新塑造（Reshape）为 $h$ 个头：$\mathbb{R}^{h \times N \times d_k}$（由于仅调整内存指针/视图，Shape 变换的开销为 $O(1)$ 或可忽略）。
* **计算量：**
3 次 $(N \times d) \times (d \times d)$ 的矩阵乘法：

$\text{FLOPs}_1 = 3 \times (N \cdot d \cdot d) = 3 N d^2$



---

#### 步骤 2：计算 $h$ 个头的注意力打分矩阵 ($Q_i \cdot K_i^T$)

* **计算过程：**
对于第 $i$ 个头（$i = 1, \dots, h$）：
$Q_i \in \mathbb{R}^{N \times d_k}$ 与 $K_i^T \in \mathbb{R}^{d_k \times N}$ 相乘，得到该头的注意力得分矩阵 $A_i \in \mathbb{R}^{N \times N}$。
* **单个头的计算量：**
$(N \times d_k)$ 与 $(d_k \times N)$ 矩阵相乘：

$\text{FLOPs}_{\text{single}} = N \cdot d_k \cdot N = N^2 \cdot d_k$


* **$h$ 个头的总计算量：**
将 $d_k = \frac{d}{h}$ 代入：

$\text{FLOPs}_2 = h \times (N^2 \cdot d_k) = h \times \left(N^2 \cdot \frac{d}{h}\right) = N^2 \cdot d$



---

#### 步骤 3：计算 $h$ 个头的加权输出 ($A_i \cdot V_i$)

* **计算过程：**
对于第 $i$ 个头：
注意力权重 $A_i \in \mathbb{R}^{N \times N}$ 与 $V_i \in \mathbb{R}^{N \times d_k}$ 相乘，得到输出 $O_i \in \mathbb{R}^{N \times d_k}$。
* **单个头的计算量：**
$(N \times N)$ 与 $(N \times d_k)$ 矩阵相乘：

$\text{FLOPs}_{\text{single}} = N \cdot N \cdot d_k = N^2 \cdot d_k$


* **$h$ 个头的总计算量：**
同样将 $d_k = \frac{d}{h}$ 代入：

$\text{FLOPs}_3 = h \times (N^2 \cdot d_k) = h \times \left(N^2 \cdot \frac{d}{h}\right) = N^2 \cdot d$



---

#### 步骤 4：多头拼接（Concat）与最终线性变换

* **计算过程：**
将 $h$ 个头的输出 $O_1, O_2, \dots, O_h$ 拼接恢复成原维度 $O \in \mathbb{R}^{N \times d}$（仅内存拼接，无乘法开销）。
然后乘以输出权重矩阵 $W^O \in \mathbb{R}^{d \times d}$。
* **计算量：**
$(N \times d)$ 与 $(d \times d)$ 矩阵相乘：

$\text{FLOPs}_4 = N \cdot d \cdot d = N d^2$



---

### 汇总与对比

<img width="694" height="278" alt="image" src="https://github.com/user-attachments/assets/0b3545ad-1e09-448e-ba41-f2d7dca78554" />


转换为大 O 表示法（忽略常数项）：


$\mathbf{O(N^2 \cdot d + N \cdot d^2)}$

#### 对比总结：

| 机制 | 单头注意力 (Single-Head) | 多头注意力 (Multi-Head) |
| --- | --- | --- |
| **维数分配** | 1 个头，维度 $d$ | $h$ 个头，每个头维度 $d_k = \frac{d}{h}$ |
| **Q·Kᵀ 计算量** | $N^2 \cdot d$ | $h \times \left(N^2 \cdot \frac{d}{h}\right) = N^2 \cdot d$ |
| **A·V 计算量** | $N^2 \cdot d$ | $h \times \left(N^2 \cdot \frac{d}{h}\right) = N^2 \cdot d$ |
| **总时间复杂度** | $\mathbf{O(N^2 \cdot d + N \cdot d^2)}$ | $\mathbf{O(N^2 \cdot d + N \cdot d^2)}$ |

### 核心结论与面试加分点

1. **数学等价性：** 分成 $h$ 个头相当于把一个维度为 $d$ 的全量计算，拆分为 $h$ 个维度为 $\frac{d}{h}$ 的子计算，**计算量在 $h$ 项上被精确抵消**。
2. **多头的本质作用：** 多头机制的本质是**在不增加计算开销的前提下，增强模型的特征提取表达能力**。它允许模型同时在不同的子空间（Subspaces）中关注不同位置、不同层次的语义关联（例如：一个头关注语法关系，另一个头关注代词指代等）。


## 既然多头注意力计算量一样，为什么大语言模型（LLM）推理时要引入 GQA (Grouped-Query Attention) 和 MQA (Multi-Query Attention)？它们优化了什么？

在大语言模型（LLM）的**推理（Inference）**阶段，影响生成速度和吞吐量的核心瓶颈往往不是“计算量（FLOPs）”，而是**“内存带宽（Memory Bandwidth）”**。

引入 MQA 和 GQA，本质上是为了**极大地减少推理时的 KV Cache（键值缓存）体积与显存读取开销**，打破“访存受限（Memory-Bound）”的瓶颈。

---

### 一、 背景：推理时的核心痛点 —— KV Cache

在自回归（Auto-Regressive）逐字生成文本时，为了避免重复计算历史 Token 的 $K$ 和 $V$ 矩阵，模型会将之前所有 Token 的 $K$ 和 $V$ 存入 GPU 显存中，这就是 **KV Cache**。

* **KV Cache 的显存开销公式：**

$\text{KV Cache 大小} = 2 \times \text{层数} \times \text{序列长度 } N \times \text{特征维度 } d \times \text{批大小 (Batch Size)}$



在多头注意力（MHA）中，每个头都有独立的一套 $K$ 和 $V$。
例如，一个 70B 参数的模型（如 Llama 2 70B，80 层，84 个头，隐藏层维度 $d=8192$），在上下文长度为 4096、Batch Size 为 32 时：

* 仅仅保存 **KV Cache 就会占用超过 100 GB 的显存**！

更为严重的是，在生成阶段（Decode 阶段），GPU 每生成一个新 Token，都需要将这上百 GB 的历史 KV Cache 从 HBM（高带宽显存）重新加载到 SRAM（高速缓存/计算单元）中。由于计算量极小（仅计算 1 个 Token），GPU 算力大量闲置，**生成速度完全取决于显存读取的速度**。

---

### 二、 架构对比：MHA、MQA 与 GQA

为了降低 KV Cache 的大小，研究人员对 $Q, K, V$ 头数的对应关系进行了重构：

| 注意力变体 | Query 头数 ($N_q$) | Key/Value 头数 ($N_{kv}$) | KV 共享机制 |
| --- | --- | --- | --- |
| **MHA** (Multi-Head) | $h$ 个 | $h$ 个 | 每个 Query 头有**专属**的 $K, V$ 头（1:1） |
| **MQA** (Multi-Query) | $h$ 个 | **1 个** | **所有** Query 头共享**同 1 个** $K, V$ 头（$h$:1） |
| **GQA** (Grouped-Query) | $h$ 个 | $g$ 个 ($1 < g < h$) | Query 头分组，每组共享 1 个 $K, V$ 头（$h/g$:1） |

#### 1. MQA (Multi-Query Attention)

* **设计：** 只有 1 个 Key 头和 1 个 Value 头。
* **优势：** KV Cache 体积直接缩减为原来的 **$\frac{1}{h}$**（例如从 32 个头缩减到 1 个，显存占用暴跌 96.8%）。显存带宽瓶颈大幅缓解，推理吞吐量提升数倍，能够支持极大的 Batch Size。
* **劣势：** 模型表达能力受损较重，在某些复杂语义建模任务中准确率下降明显，且训练容易不稳定。

#### 2. GQA (Grouped-Query Attention)

* **设计：** 将 $h$ 个 Query 头分为 $g$ 个组（如 8 组），每组内的 Query 头共享同一个 $K, V$ 头。
* **优势：** **在性能与效果之间取得了极佳平衡**。
* KV Cache 减少为原来的 **$\frac{g}{h}$**（例如 8 组时，KV Cache 减少 75%-87%）。
* 效果几乎**完全追平 MHA**，同时吞吐量和推理速度接近 MQA。



---

### 三、 它们究竟优化了什么？

#### 1. 显存容量与 KV Cache 占用（最直接优化）

* 显著降低了单个请求对 GPU 显存的占用。使得在有限的显存（如单张 A100/H100）下，模型能够支持**更长的上下文（Long Context）**以及**更大的并发量（Batch Size）**。

#### 2. 内存带宽与推理延迟（生成速度优化）

* 在 Decode 阶段，GPU 计算是典型的 **Memory-Bound（访存受限）** 任务。KV Cache 变小意味着 GPU 从 HBM 加载数据到 Tensor Core 的字节数大幅减少，每个 Token 的生成延迟（Time-Per-Output-Token）显著降低。

#### 3. 计算量（FLOPs）变化微乎其微

* 需要注意：在 Prefill（首字填充）阶段，因为 $Q$ 的头数依旧是 $h$ 个，计算注意力打分矩阵（$Q \cdot K^T$）时的运算量依然是 $O(N^2 \cdot d)$ 级别，计算量只在生成 $K, V$ 的线性投影层略微减少，**总体计算量与 MHA 基本持平**。

---

### 四、 总结与现状

| 维度 | MHA (如 Llama 1 / GPT-3) | MQA (如 Falcon / PaLM) | GQA (如 Llama 2 70B / Llama 3 / Mistral) |
| --- | --- | --- | --- |
| **KV Cache 显存大小** | 100% | **$\frac{1}{h}$ (最低)** | **$\frac{g}{h}$ (极低)** |
| **推理速度与吞吐量** | 基准（容易受限于显存带宽） | 极快（最高） | 极快（接近 MQA） |
| **模型准确率与稳定性** | 最佳 | 有明显衰减风险 | **几乎无损（接近 MHA）** |

**业界现状：**

目前 **GQA** 已经成为大语言模型（如 Llama 3、Mistral、Qwen 等）的**标准配置**。它通过重构注意力头数，以极微小的效果代价，解决了 LLM 在实际落地部署时的显存与推理吞吐瓶颈。

## 注意力计算优化有哪些技术

在大语言模型（LLM）向长上下文、大并发部署演进的过程中，**注意力计算（Self-Attention）**的优化成为了提升性能的关键。针对自注意力机制在**计算复杂度 $O(N^2)$**、**显存空间占用**以及推理时的内存带宽瓶颈（Memory-Bound）等问题，业界演进出了多维度的技术路线：

---

### 一、 架构与头数重构（减少 KV Cache 显存与访存）

这类技术主要作用于**推理 Decode 阶段**，通过减少 Key/Value 的头数来大幅降低 KV Cache 显存占用，从而提高推理吞吐量。

* **MQA (Multi-Query Attention)：** 所有 Query 头共享同 1 个 Key/Value 头。KV Cache 体积缩减至 $\frac{1}{h}$，大幅降低显存带宽压力，但对模型表达能力有一定损伤。
* **GQA (Grouped-Query Attention)：** 将 Query 头分组（如 8 组），每组共享 1 个 Key/Value 头。在 KV Cache 开销与模型效果之间取得了最佳平衡，已成为 Llama 3、Qwen 等主流开源大模型的标配。
* **MLA (Multi-Head Latent Attention)：** DeepSeek V2/V3 提出的创新架构。利用低秩矩阵分解（Low-Rank Compression），将 KV Cache 压缩映射到一个低维隐空间中，在保留多头表达能力的同时，实现了比 GQA 更低的 KV Cache 显存占用。

---

### 二、 硬件级与算子融合优化（打破 Memory-Bound 瓶颈）

这类技术不改变注意力的数学输出，而是利用 GPU 存储层级结构（HBM 与 SRAM）减少内存读写，实现精确的高效计算。

* **FlashAttention (v1/v2/v3)：**
* **平铺（Tiling）：** 将长序列的 $Q, K, V$ 矩阵切分为小 Block 加载到高速 SRAM 中计算。
* **在线 Softmax（Online Softmax）：** 避免在 HBM 中存储巨大的 $N \times N$ 注意力得分矩阵，将显存空间复杂度从 $O(N^2)$ 降低到 **$O(N)$**。
* **Recomputation（反向传播重算）：** 训练时不保存中间注意力矩阵，反向传播时在 SRAM 中重新计算，大幅节省显存。


* **PagedAttention (vLLM)：** 借鉴操作系统虚拟内存的分页思想，将连续的 KV Cache 离散化存储在不连续的显存块（Pages）中，彻底消除了显存碎片化，将显存利用率提升至接近 100%。

---

### 三、 稀疏化与线性注意力（降低算法复杂度到 $O(N)$）

通过近似计算或稀疏采样，直接降低长序列下的计算量。

#### 1. 稀疏注意力（Sparse Attention）

* **固定/局部窗口注意力（Local / Window Attention）：** 每个 Token 仅与周围固定窗口内的 Token 发生交互（如 Longformer、Mistral 划窗注意力 Sliding Window Attention）。
* **动态/动态稀疏注意力（Dynamic / Sparse Pattern）：** 如 Reformer（使用局部敏感哈希 LSH 分桶）、Routing Transformer 等，仅计算关联度最高的一部分 Token 对。

#### 2. 线性注意力（Linear Attention）与状态空间模型

* **核函数近似（Kernelization）：** 将 $\text{Softmax}(QK^T)V$ 通过核函数拟合拆解为 $Q(K^TV)$，将计算顺序改变后将复杂度降至 $O(N \cdot d^2)$。
* **新型 RNN / 状态空间模型（SSM）：** 如 **Mamba / RWKV**，抛弃传统的 $N \times N$ 全矩阵计算，采用循环递推（Recurrence）的形式将历史信息压缩在固定大小隐状态中，实现线性的推理与训练开销。

---

### 四、 序列并行与长上下文分布式优化

当单张 GPU 无法容纳超长上下文的注意力计算时，采用分布式切分技术：

* **Ring Attention：** 将长序列按 Token 维度切分到多张 GPU 上，卡与卡之间组成环形通信（Ring Communication），在重叠（Overlap）计算与通信的同时完成全局注意力计算，使上下文长度能够随着 GPU 数量呈线性扩展。
* **Context Parallelism (CP)：** 将序列切分与 FlashAttention 算子结合，在多卡间并行处理 Long Context。

---

### 总结对比

| 优化技术路线 | 代表方案 | 主要解决的瓶颈 | 适用阶段 | 是否损害精度 |
| --- | --- | --- | --- | --- |
| **架构与头数重构** | GQA / MQA / MLA | 推理时的 KV Cache 显存与带宽 | 架构设计/推理 | 极其轻微/无损 |
| **算子与硬件融合** | FlashAttention / PagedAttention | 显存碎片、GPU HBM 读写延迟 | 训练 & 推理 | **完全无损 (100% 精确)** |
| **稀疏/线性化** | Sliding Window / Mamba | 长文本计算量 $O(N^2)$ 爆炸 | 架构设计 | 存在一定近似损失 |
| **分布式序列并行** | Ring Attention | 单卡显存无法容纳超长文本 | 大规模训练/推理 | **完全无损** |

## 请详细解释 FlashAttention 中使用的 Online Softmax 算法原理，以及它是如何通过 Tiling 减少 HBM 读写的？
FlashAttention 的核心突破在于：在**不改变标准注意力计算结果（完全无损）**的前提下，将注意力计算的**显存空间复杂度从 $O(N^2)$ 降低到了 $O(N)$**，同时大幅减少了 GPU 硬件上的内存读写延迟。

这一突破的核心正是依靠平铺（Tiling）**与**在线 Softmax（Online Softmax）两大技术的结合。
<img width="855" height="359" alt="image" src="https://github.com/user-attachments/assets/29a6dc3e-1748-478b-8199-5d8b582ad4e2" />

---

### 一、 核心痛点：为什么标准 Attention 会受限于 HBM 读写？

GPU 的存储体系分为两层：

1. **HBM（高带宽显存，容量大但速度慢）：** 比如 A100 的 80GB 显存，带宽约 2 TB/s。
2. **SRAM（片上高速缓存，容量极小但速度极快）：** 比如 A100 每张卡只有约 192 KB / Streaming Multiprocessor，带宽高达 19 TB/s（接近 HBM 的 10 倍）。

在标准的 Self-Attention 计算中：


$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

1. 计算 $S = QK^T$，将大小为 $N \times N$ 的矩阵 $S$ **写回 HBM**；
2. 从 HBM 读出 $S$，计算 $P = \text{Softmax}(S)$，再将大小为 $N \times N$ 的 $P$ **写回 HBM**；
3. 从 HBM 读出 $P$ 和 $V$，计算 $O = PV$，将 $O$ **写回 HBM**。

**问题所在：**

对于长文本（例如 $N=64k$），$N \times N$ 的注意力矩阵非常巨大，无法存入极小的 SRAM，必须频繁在 HBM 和 SRAM 之间搬运数据。由于 HBM 读写速度远远跟不上 Tensor Core 的计算速度，GPU 的大部分时间都在等待数据传输，处于 **Memory-Bound（访存受限）** 状态。

---

### 二、 Tiling（分块/平铺）技术：打破内存墙

FlashAttention 的思想是：**绝不将巨大的 $N \times N$ 注意力得分矩阵落盘到 HBM 中**。

1. **分块加载：** 将 $Q$ 矩阵切分为大小为 $B_r \times d$ 的小块（Block），将 $K, V$ 矩阵切分为大小为 $B_c \times d$ 的小块。
2. **片上计算：** 将小块加载到高速 SRAM 中，在 SRAM 内完成小块的矩阵乘法和 Softmax 累加。
3. **分块写回：** 最终只将大小为 $N \times d$ 的最终输出 $O$ 写回 HBM。

但是，Tiling 遇到了一个数学障碍：**Softmax 是一个“全局”函数**。

标准 Softmax 公式为：


$P_{ij} = \frac{e^{S_{ij} - m}}{\sum_{k=1}^N e^{S_{ik} - m}} \quad (\text{其中 } m = \max_k S_{ik} \text{ 用于数值稳定})$

按分块加载时，只能拿到局部数据，**在没看到完整序列前，既无法知道全局最大值 $m$，也无法算出全局分母（指数和）$l$**。这就需要引入 **Online Softmax**。

---

### 三、 Online Softmax 算法原理与数学推导

Online Softmax（在线 Softmax）允许我们在**遍历序列的过程中，流式（Stream）地动态更新 Softmax 的局部分子和分母**，从而实现分块增量计算。

为了直观理解，假设一个行向量 $S$ 被切分为两块：$S = [S^{(1)}, S^{(2)}]$。

#### 1. 块 1 的局部计算（Block 1）

计算第一块 $S^{(1)}$ 的局部最大值 $m^{(1)}$ 和局部指数和 $l^{(1)}$：

* $m^{(1)} = \max(S^{(1)})$
* $l^{(1)} = \sum e^{S^{(1)} - m^{(1)}}$
* 局部输出乘积：$O^{(1)} = e^{S^{(1)} - m^{(1)}} V^{(1)}$

#### 2. 结合块 2 进行增量更新（Block 2）

当拿到第二块 $S^{(2)}$ 时，计算第二块的局部最大值 $m^{(2)}$ 和局部指数和 $l^{(2)}$：

* $m^{(2)} = \max(S^{(2)})$
* **更新全局最大值：** $m^{\text{new}} = \max(m^{(1)}, m^{(2)})$

因为最大值修正了，之前基于 $m^{(1)}$ 算出的指数项需要乘以缩放因子 $e^{m^{(1)} - m^{\text{new}}}$ 进行修正：

<img width="703" height="270" alt="image" src="https://github.com/user-attachments/assets/d4887fae-5395-4c95-b42f-5a2e8d679b92" />



#### 3. 最终归一化

当遍历完所有分块后，只需要将累加结果除以最终的全局分母 $l^{\text{final}}$：


$O = \frac{O^{\text{final}}}{l^{\text{final}}}$

通过这种**修正因子（Correction Factor）**机制
$e^{m^{\text{old}} - m^{\text{new}}}$ 
，FlashAttention 可以完全在 SRAM 中一步步更新中间统计量 $(m, l, O)$ ，**完全不需要在显存里保存中间的 $N \times N$ 注意力得分矩阵**。

---

### 四、 总结：FlashAttention 带来的三重红利

1. **显存占用骤降：** 不再存储 $N \times N$ 的注意力矩阵，显存空间复杂度从 **$O(N^2)$ 降低到 $O(N)$**，使得单卡训练超长上下文（如 32k、128k）成为可能。
2. **计算大幅加速：** 虽然 Online Softmax 增加了一些额外的标量指数运算（约增加 15-20% 的算术开销），但由于**把 HBM 读写次数降低了一个数量级**，整体运行速度提升了 **2 - 4 倍**（以计算换带宽）。
3. **精细化控制：** 巧妙配合反向传播时的重算（Recomputation），反向传播时也不保存 $N \times N$ 矩阵，而是根据 $Q, K, V$ 和保存的统计量 $(m, l)$ 在 SRAM 中重新生成，进一步省下了大量显存。

## pagedAttention

在大模型（LLM）推理的 Decode（逐字生成）阶段，显存资源是决定系统的**并发吞吐量（Throughput）**和**最大上下文长度**的关键瓶颈。

传统的 KV Cache 管理机制存在严重的显存浪费，**vLLM** 团队因此提出了 **PagedAttention**，借鉴操作系统中的虚拟内存与分页（Virtual Memory Paging）机制，从根本上解决了这一问题。

---

### 一、 动机：传统 KV Cache 的三大显存痛点

在传统的 LLM 推理框架中，为了保证矩阵乘法（GEMM）在连续内存地址上的高效计算，系统必须为每个请求在显存中开辟**一块连续的内存空间**来存放 KV Cache。

这种“连续预分配”模式带来了三类极其严重的浪费：

1. **预保留浪费（Over-reservation）：**
模型不知道请求最终会输出多少个 Token（例如预设 `max_len = 2048`），因此必须按最大长度提前预留连续显存。如果用户只生成了 10 个字，剩下的 2000 个 Token 的显存空间就完全被闲置了。
2. **碎片化浪费（Fragmentation）：**
在多并发场景下，频繁的请求创建与销毁会导致显存出现大量不连续的小空闲块（外部碎片），无法凑出一块完整的连续空间来容纳新的长文本请求。
3. **共享困难（Sharing Loss）：**
在 Parallel Sampling（一问多答）、Beam Search 或带有很长 System Prompt（系统提示词）的场景下，多个请求的 Prefix（前缀）内容完全相同，但传统方式不得不为每个请求重复复制多份相同的 KV Cache。

> **统计数据：** 传统框架中，**真正有效用于存储 KV Cache 的显存往往只有 20%~40%**，其余 60%+ 的显存都被预保留和碎片浪费掉了。

---

### 二、 改进措施：借鉴操作系统的“分页机制”

PagedAttention 的核心思想是：**打破 KV Cache 在物理显存中必须连续存储的限制**。

如同现代操作系统通过页表（Page Table）将连续的虚拟内存映射到离散的物理内存（RAM）一样，vLLM 实现了：

* **逻辑上连续，物理上离散：** 允许 KV Cache 分散存储在不连续的显存块（Physical Blocks）中。
* **按需分配（On-Demand Allocation）：** 不再提前预留空间，生成一个 Token 就按需分配一块显存，不够用了再动态申请新的 Block。
* **块级共享（Block-level Sharing）：** 多个请求包含相同前缀时，只需映射到同一个物理块，实现零拷贝共享（Copy-on-Write）。

---

### 三、 工作原理与计算示例

#### 1. 核心数据结构

* **Block（物理块）：** 显存中一块固定大小的连续空间（比如可容纳 $B=16$ 个 Token 的 KV Cache）。
* **Logical Block（逻辑块）：** 序列在逻辑上按 16 个 Token 一组划分为逻辑块 $0, 1, 2 \dots$。
* **Block Table（映射页表）：** 记录逻辑块与物理显存块（Physical Block）的映射关系。

#### 2. 注意力计算过程（PagedAttention Kernel）

在计算 Attention 时，GPU Kernel 不再从连续地址读取 $K, V$，而是依据 Block Table 的指引：


$$\text{Attention 得分 } S_i = Q_i \cdot K_{\text{Block\_Table}[j]}^T$$


通过自定义的 CUDA Kernel，在离散的物理块之间高效地执行 Gather（收集）与 Scatter（分散）计算，实现与连续内存完全一致的数学结果。

---

#### 3. 动态分配与共享示例

假设块大小（Block Size）$B = 4$（即每个 Block 存 4 个 Token）。

##### 场景 A：动态增长（按需分配）

一个请求开始生成文本，当前生成了 6 个 Token：

1. **逻辑 Token 0~3：** 被分配到 **物理块 Block 7**（已被填满）。
2. **逻辑 Token 4~5：** 被分配到 **物理块 Block 12**（目前只用了 2 个位置，剩余 2 个可用）。
3. **生成第 7 个 Token 时：** 直接写入 **物理块 Block 12** 的第 3 个空位。
4. **生成第 9 个 Token 时（Block 12 已满）：** 系统动态从物理显存池中申请一个全新的 **Block 3** 续接。

没有任何提前的虚高预留，显存开销精确匹配当前生成的实际长度。

##### 场景 B：前缀共享（Copy-on-Write）

假设输入了相同的 Prompt（8 个 Token，对应逻辑块 0 和 1）：

```
[请求 A 页表]                    [物理显存 Block]
逻辑块 0 ──────────────┐       ┌──> Block 5 (Prompt 前 4 个 Token, 引用计数=2)
逻辑块 1 ────────┐     └───────┼──> Block 9 (Prompt 后 4 个 Token, 引用计数=2)
                 │             │
[请求 B 页表]    │             │
逻辑块 0 ────────┼─────────────┘
逻辑块 1 ────────┘

```

1. 请求 A 和请求 B 的逻辑块 0 和 1 **共同指向物理 Block 5 和 Block 9**。
2. 当请求 A 生成新的 Token 需要写入时，触发**写时复制（Copy-on-Write）**：只为请求 A 分配一个新的物理 Block 15 写入新数据，而相同的 Prompt 显存依然被 A 和 B 共享。

---

### 四、 总结收益

通过 PagedAttention 机制，vLLM 取得了显著的性能提升：

* **显存浪费率从 60%+ 降低到 < 4%**（仅剩每个序列最后一个 Block 内极其轻微的内部碎片）。
* **并发吞吐量（Throughput）提升 2~4 倍**：节省出的显存被用于扩大并发 Batch Size，大幅提升了 GPU 的张量计算利用率。
* **支持长文本与复杂抽样**：Beam Search 和长系统提示词场景下的显存开销得到数量级下降。

## 对比flashattention和pageAttention

**FlashAttention** 和 **PagedAttention** 都是大语言模型（LLM）领域里程碑式的注意力机制优化技术，但它们的**设计初衷、解决的核心痛点以及作用阶段完全不同**。

简单来说：

* **FlashAttention 解决的是“算得慢/算不动”的问题**：它是一个**计算算子（Kernel）优化**，通过利用 GPU 高速缓存（SRAM）分块计算，避开慢速显存（HBM）的读写瓶颈，使得长上下文的计算速度大幅提升。
* **PagedAttention 解决的是“显存不够/存不下”的问题**：它是一个**系统级显存管理**优化，通过借鉴操作系统的分页机制，消除 KV Cache 的显存碎片与预留浪费，大幅提升系统的并发吞吐量。

---

### 一、 核心维度对比表

| 对比维度 | FlashAttention (v1 / v2 / v3) | PagedAttention (vLLM) |
| --- | --- | --- |
| **技术本质** | 硬件感知的 **CUDA 算子融合与并行算法** | 借鉴操作系统虚拟内存的 **显存管理机制** |
| **解决的核心痛点** | 显存带宽限制（Memory-Bound）、$N^2$ 显存空间开销 | 显存碎片化、预分配浪费、多并发前缀共享困难 |
| **优化目标** | 提升**单个请求的计算速度**与长文本处理能力 | 提升**系统整体的并发吞吐量（Throughput）** |
| **作用阶段** | **训练（Training）** 与 **推理（Inference）** 均适用 | 主要适用于 **推理 Decode 阶段** |
| **核心算法/技术** | **Tiling（分块）** + **Online Softmax** | **Virtual Memory Paging（分页映射表）** |
| **对模型精度的影响** | **完全无损**（数学上与标准 Attention 严格等价） | **完全无损**（仅改变物理存储，计算等价） |
| **代表应用框架** | PyTorch (`F.scaled_dot_product_attention`)、Megatron-LM | **vLLM**、TensorRT-LLM、SGLang |

---

### 二、 核心原理与机制差异

#### 1. FlashAttention：重构计算流程，打破“内存墙”

* **背景痛点：** 标准 Attention 需要计算并保存 $N \times N$ 的注意力矩阵。当序列长度 $N$ 变长时，这个矩阵巨大，必须频繁在 GPU 的主显存（HBM）**和**片上高速缓存（SRAM）之间来回传输数据。由于 HBM 带宽有限，GPU 的计算单元大部分时间都在等待数据加载（Memory-Bound）。
* **改进机制：**
1. **分块（Tiling）：** 将 $Q, K, V$ 矩阵切小，直接加载进极快的 **SRAM** 中计算。
2. **Online Softmax：** 巧妙地在流式遍历数据时动态修正 Softmax 的最大值和分母，**完全不向 HBM 写入中间的 $N \times N$ 得分矩阵**。


* **效果：** 显存占用从 $O(N^2)$ 降为 $O(N)$，读写次数大幅减少，计算速度提升 **2~4 倍**。

#### 2. PagedAttention：重构显存分配，消除“显存碎片”

* **背景痛点：** 推理生成文本时，由于无法预知用户会生成多少字，传统框架必须按最大可能长度（如 2048）提前开辟**连续**的 KV Cache 显存空间。这导致 60% 以上的显存被虚高预留和碎片化浪费，严重限制了并发量。
* **改进机制：**
1. **物理离散，逻辑连续：** 将 KV Cache 切分为固定大小的 Block（如每个 Block 存 16 个 Token）。
2. **按需分配与页表映射：** 生成一个 Token 分配一点空间，使用 **Block Table（页表）** 记录逻辑位置到物理显存的映射。
3. **零拷贝共享：** 多个请求遇到相同系统提示词（Prompt）时，直接映射到同一物理 Block（Copy-on-Write）。


* **效果：** 显存浪费率从 >60% 降至 <4%，单个 GPU 能容纳的**并发 Batch Size 提升 2~4 倍**。

---

### 三、 经典物理比喻

如果把 GPU 显存比作 **酒店的房间**，计算过程比作 **服务员清理房间**：

* **FlashAttention 就像是优化“服务员的操作流程”**：
以前服务员洗一件衣服（计算一步）就要跑一趟地下仓库（HBM）；现在服务员带了一个高效的小推车（SRAM），在房间门口集中洗完推进去，不再频繁跑腿，**干活速度变快了**。
* **PagedAttention 就像是改进“酒店的订房系统”**：
以前客人（请求）来订房，哪怕只住 1 天，酒店也必须一次性预留连续 30 天的大套房，导致大量空房无法卖给别人；现在改为按天、按单间动态分配（分页），允许客人散落在不同楼层，**酒店接客的并发吞吐量翻倍了**。

---

### 四、 现代大模型推理框架中的协同结合

在实际的生产环境（如 vLLM、TensorRT-LLM、SGLang 等现代推理引擎）中， FlashAttention 与 PagedAttention 并不是竞争关系，而是深度结合的：

1. **PagedAttention 管理显存：** 在宏观上，利用 PagedAttention 的 Block Table 去离散地管理、组织和调度所有请求的 KV Cache，最大化显存利用率。
2. **FlashAttention 执行计算：** 在微观上，当 CUDA Kernel 执行注意力计算时，底层融合了 FlashAttention 的算法思想（如 FlashDecoding / Paged-FlashAttention），直接从 PagedAttention 的离散 Block 中分块加载 $K, V$ 到 SRAM 进行高性能的流式计算。

两者结合，既保证了**单请求计算的高速度（FlashAttention）**，又保证了**多请求并发的高吞吐（PagedAttention）**。



