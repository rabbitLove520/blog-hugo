---
title: Attention机制学习
author: pigLoveRabbit
tags:
  - llm
categories:
  - 深度学习
date: 2024-12-01 20:00:00
---
![logo](/images/attention_formula.png)

<!-- more -->

## Q K V
在深度学习中，很多 LLM 的训练都使用 Transformer 架构，而在 Transformer 架构中计算的过程涉及到的最关键的就是注意力，它是整个过程中重要的基础。注意力抽象出了 3 个重要的概念，在计算过程中对应着 3 个矩阵，如下所示：

* Query：在自主提示下，自主提示的内容，对应着矩阵 Q
* Keys：在非自主提示下，进入视觉系统的线索，对应着矩阵 K
* Values：使用 Query 从 Keys 中匹配得到的线索，基于这些线索得到的进入视觉系统中焦点内容，对应着矩阵 V

我们要训练的模型，输入的句子有 n 个 token，而通过选择并使用某个 Embedding 模型获取到每个 token 的 Word Embedding，**每个 Word Embedding 是一个 d 维向量**。本文我们详细说明自注意力（Self-Attention）的计算过程，在进行解释说明之前，先定义一些标识符号以方便后面阐述使用：


KV Cache（Key-Value Cache）是大语言模型（LLM）推理过程中至关重要的一项优化技术。它直接决定了模型生成的**速度（延迟）**和**显存占用**。

以下是对 KV Cache 的详解，涵盖其背景、原理、显存计算、优化方案及挑战。

---

## KV Cache

要理解 KV Cache，首先需要理解 Transformer 的 **Self-Attention（自注意力）** 机制和 **Autoregressive（自回归）** 生成模式。

#### 1.1 Self-Attention 机制
在 Transformer 中，注意力计算公式为：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
其中：
*   $Q$ (Query): 当前 token 想要查询的信息。
*   $K$ (Key): 所有 token 的特征标识。
*   $V$ (Value): 所有 token 的实际内容信息。

#### 1.2 自回归生成的冗余
LLM 生成文本是逐个 token 进行的（例如：输入 "I love"，输出 "AI"）。
*   **第 1 步**：输入 "I"，计算 $Q_1, K_1, V_1$，输出概率。
*   **第 2 步**：输入 "I love"，计算 $Q_2, K_2, V_2$。**注意**：为了计算 "love" 对 "I" 的注意力，模型需要再次计算 "I" 的 $K_1, V_1$。
*   **第 3 步**：输入 "I love AI"，计算 $Q_3, K_3, V_3$。模型需要再次计算 "I" 和 "love" 的 $K, V$。

**问题**：如果没有缓存，每一步生成新 token 时，都要重新计算之前所有 token 的 $K$ 和 $V$ 矩阵。随着序列长度 $N$ 增加，计算量呈 $O(N^2)$ 增长，造成巨大的算力浪费。

**解决方案**：**KV Cache**。将之前计算好的 $K$ 和 $V$ 矩阵保存在显存中，生成新 token 时直接读取，只计算新 token 的 $Q, K, V$。

---

### 2. KV Cache 的工作原理

推理过程分为两个阶段：**Prefill（预填充）** 和 **Decoding（解码）**。

#### 2.1 Prefill 阶段（处理 Prompt）
*   **输入**：完整的提示词（Prompt），例如长度为 $L$ 的序列。
*   **操作**：模型并行处理所有 $L$ 个 token。
*   **缓存**：计算完所有层的 $K$ 和 $V$ 后，将其存入 KV Cache。
*   **输出**：生成第一个新 token 的概率分布。
*   **特点**：计算密集型（Compute-bound），主要消耗 GPU 算力。

#### 2.2 Decoding 阶段（逐个生成）
*   **输入**：上一步生成的 1 个新 token。
*   **操作**：
    1.  计算新 token 的 $q, k, v$（小写表示单 token 向量）。
    2.  **读取缓存**：从 KV Cache 中取出之前 $L$ 个 token 的 $K, V$ 矩阵。
    3.  **拼接**：将新的 $k, v$ 拼接到缓存的 $K, V$ 后面，形成完整的 $K_{new}, V_{new}$。
    4.  **注意力计算**：使用 $q$ 和 $K_{new}, V_{new}$ 计算注意力。
    5.  **更新缓存**：将新的 $k, v$ 写入 KV Cache，供下一步使用。
*   **特点**：显存带宽密集型（Memory-bound），主要消耗显存带宽。

#### 2.3 图解流程
```text
Step 1 (Prompt): [Token1, Token2] -> 计算 K1, V1, K2, V2 -> 存入 Cache
Step 2 (Gen 1):  [Token3]      -> 计算 k3, v3 
                               -> 读取 Cache (K1, V1, K2, V2)
                               -> 拼接 -> 计算 Attention -> 输出 Token3
                               -> 更新 Cache (存入 k3, v3)
Step 3 (Gen 2):  [Token4]      -> 计算 k4, v4
                               -> 读取 Cache (K1...V3)
                               -> ...
```

---

### 3. KV Cache 显存占用计算

KV Cache 是长文本推理时的主要显存瓶颈。了解其计算公式对于评估硬件需求至关重要。

#### 3.1 计算公式
$$ \text{Memory} = 2 \times \text{Num\_Layers} \times \text{Hidden\_Size} \times \text{Seq\_Len} \times \text{Batch\_Size} \times \text{Bytes\_Per\_Param} $$

*   **2**: 因为要存 Key 和 Value 两份。
*   **Num\_Layers**: 模型层数（如 Llama-7B 为 32 层）。
*   **Hidden\_Size**: 隐藏层维度（如 Llama-7B 为 4096）。
    *   *注：更精确的是 `Num_Heads * Head_Dim`，通常等于 Hidden_Size。*
*   **Seq\_Len**: 序列总长度（Prompt 长度 + 生成长度）。
*   **Batch\_Size**: 并发请求数。
*   **Bytes\_Per\_Param**: 数据类型精度（FP16/BF16 = 2 字节，INT8 = 1 字节，FP8 = 1 字节）。

#### 3.2 实例计算 (Llama-7B, FP16)
假设 Batch Size = 1，序列长度 = 4096。
*   层数：32
*   隐藏层：4096
*   精度：2 Bytes (FP16)

$$ \text{Memory} = 2 \times 32 \times 4096 \times 4096 \times 1 \times 2 \text{ Bytes} $$
$$ \approx 2,147,483,648 \text{ Bytes} \approx 2 \text{ GB} $$

**结论**：
*   Llama-7B 模型权重约占 14GB (FP16)。
*   如果生成长度达到 32k tokens，KV Cache 占用将高达 **16GB**。
*   这意味着在长上下文场景下，**KV Cache 的显存占用甚至可能超过模型权重本身**。

---

### 4. KV Cache 优化技术

为了减少显存占用并提高吞吐量，业界提出了多种优化方案。

#### 4.1 多查询注意力 (MQA) 与 分组查询注意力 (GQA)
*   **MHA (Multi-Head Attention)**: 标准模式，Q、K、V 的头数相同（如 32 头）。显存占用大。
*   **MQA (Multi-Query Attention)**: Q 保持多头，但 K 和 V 只有 **1 个头**。所有 Q 头共享同一组 K/V。
    *   *效果*: KV Cache 显存减少 32 倍。
    *   *缺点*: 模型精度可能轻微下降。
*   **GQA (Grouped-Query Attention)**: 折中方案。Q 分为 32 头，K/V 分为 8 组（每组 4 个 Q 头共享）。
    *   *效果*: KV Cache 显存减少 4 倍，精度接近 MHA。
    *   *应用*: Llama-2-70B, Llama-3 等主流模型均采用 GQA。

#### 4.2 KV Cache 量化 (Quantization)
将 KV Cache 从 FP16 压缩到 INT8 甚至 FP4。
*   **原理**: 在存入 Cache 前进行量化，读取时反量化（或直接用量化矩阵计算）。
*   **效果**: 显存占用减少 50% (INT8) 或 75% (FP4)。
*   **挑战**: 需要保证量化带来的精度损失不影响生成质量。

#### 4.3 PagedAttention (vLLM 核心技术)
*   **问题**: 传统 KV Cache 需要在显存中分配连续内存。由于显存碎片化，往往导致“有剩余显存但无法分配”的情况，限制了并发量（Batch Size）。
*   **方案**: 借鉴操作系统的**分页内存管理**。将 KV Cache 分成非连续的 Block。
*   **优势**:
    *   消除显存碎片，显著提高显存利用率（接近 100%）。
    *   支持内存交换（Swapping），将不活跃的 KV Cache 换出到 CPU 内存。
    *   支持高效的 **Prefix Caching**（复用相同 Prompt 的 Cache）。

#### 4.4 稀疏注意力 (Sparse Attention)
*   **原理**: 并不是所有历史 token 都同等重要。只保留关键的 K/V（如最近 N 个 token + 重要的全局 token）。
*   **代表**: StreamingLLM, H2O。
*   **效果**: 将 KV Cache 大小从 $O(N)$ 降低到 $O(1)$ 或 $O(\text{constant})$，实现无限长度生成。

---

### 5. 常见挑战与问题

1.  **显存墙 (Memory Wall)**:
    在 Decoding 阶段，GPU 大部分时间在等待从显存读取 KV Cache，计算单元利用率低。优化带宽（如使用 HBM3e）比优化算力更重要。

2.  **长上下文 OOM**:
    当上下文窗口极大（如 128k, 1M）时，即使使用量化，KV Cache 也可能撑爆显存。需要结合 Offload（卸载到 CPU/磁盘）或稀疏注意力。

3.  **并发瓶颈**:
    Batch Size 越大，KV Cache 占用线性增长。这限制了服务器能同时服务的用户数量。PagedAttention 是解决此问题的关键。

4.  **动态长度管理**:
    不同请求的生成长度不同，导致 KV Cache 释放时间不一致，增加了内存管理的复杂度。

---

### 6. 总结

| 特性 | 说明 |
| :--- | :--- |
| **核心作用** | 避免自回归生成中重复计算历史 token 的 K/V 矩阵。 |
| **主要收益** | 将生成阶段的计算复杂度从 $O(N^2)$ 降为 $O(N)$，大幅提升推理速度。 |
| **主要代价** | 占用大量显存（VRAM），是长文本推理的瓶颈。 |
| **关键优化** | **GQA/MQA** (减少头数), **量化** (减少精度), **PagedAttention** (减少碎片)。 |
| **适用场景** | 所有基于 Transformer Decoder 的 LLM 推理。 |

**一句话总结**：KV Cache 是用**空间换时间**的经典策略，它是现代 LLM 能够实时对话的基石，而围绕 KV Cache 的显存优化（如 vLLM、GQA）则是提升推理吞吐量的核心战场。























* [参考](http://shiyanjun.cn/archives/2688.html)
* [Transformer图解](https://fancyerii.github.io/2019/03/09/transformer-illustrated/)
