---
title: 卷积学习
author: pigLoveRabbit
tags: []
categories:
  - python
  - 深度学习
  - pytorch
date: 2026-03-15 14:00:00
---
![upload successful](/images/triton-logo.png)  


## 卷积
[官网介绍](https://triton.hyper.ai/)：Triton 是一种用于并行编程的语言和编译器。它旨在提供一个基于 Python 的编程环境，以高效编写自定义 DNN 计算内核，并能够在现代 GPU 硬件上以最大吞吐量运行。  
简单讲，就是可以用Python写GPU算子。  
<!-- more -->
## 什么是卷积
卷积是一种提取特征的操作。

## torch.nn.Conv2d
正确理解：卷积核的通道数与数量  
首先明确核心概念：
- `in_channels`：**输入特征图的通道数**（不是“要做几次卷积”）
  - 比如RGB图片，输入通道数就是3（R、G、B各一个通道），这是输入的维度，卷积核必须和这个维度匹配才能做卷积。
- `out_channels`：**卷积核的数量**（不是“每个通道做几次卷积”）
  - 比如设置 `out_channels=6`，意味着会创建6个卷积核，每个卷积核的通道数都等于 `in_channels`（即3）。

### 直观的计算过程（以RGB图片+out_channels=6为例）
1. **单个卷积核的结构**：
   每个卷积核的维度是 `(in_channels, kernel_size, kernel_size)`，比如 `kernel_size=3` 时，单个卷积核是 `(3, 3, 3)`（3个通道，每个通道3x3的权重）。
2. **卷积计算**：
   每个卷积核会**同时对输入的3个通道做卷积**（不是“每个通道单独做”），然后把3个通道的卷积结果相加，得到1个单通道的输出特征图。
3. **多个卷积核的输出**：
   6个卷积核会得到6个单通道的输出特征图，最终拼接成一个 `6` 通道的输出特征图（维度：6×H×W）。

### 代码示例验证
用一个简单的代码直观展示通道数的变化：
```python
import torch
import torch.nn as nn

# 定义卷积层：输入3通道，输出6通道，卷积核3x3
conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)

# 模拟一张RGB图片：batch_size=1，3通道，高宽各32
input_img = torch.randn(1, 3, 32, 32)  # 维度：(batch, in_channels, H, W)

# 执行卷积
output = conv(input_img)

# 查看输出维度
print("输入维度：", input_img.shape)  # torch.Size([1, 3, 32, 32])
print("输出维度：", output.shape)    # torch.Size([1, 6, 32, 32])
print("卷积核参数维度：", conv.weight.shape)  # torch.Size([6, 3, 3, 3])
```

**关键输出解释**：
- `conv.weight.shape` 是 `(6, 3, 3, 3)`：6个卷积核，每个卷积核3通道（匹配输入），每个通道3x3权重。
- 输出维度是 `(1, 6, 32, 32)`：6个输出通道，对应6个卷积核的计算结果。

### 总结
1. `in_channels` 是**输入的通道数**（RGB图=3，灰度图=1），决定了每个卷积核的通道数；
2. `out_channels` 是**卷积核的数量**，每个卷积核生成1个输出通道，最终输出通道数等于卷积核数量；
3. 核心误区纠正：不是“每个输入通道做几次卷积”，而是“多个卷积核（数量=out_channels）分别对所有输入通道做卷积，每个卷积核输出1个通道”。
