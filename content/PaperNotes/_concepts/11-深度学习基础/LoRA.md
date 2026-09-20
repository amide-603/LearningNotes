---
type: concept
aliases: [Low-Rank Adaptation, 低秩适配]
---

# LoRA

## 定义

冻结原模型权重，仅训练低秩增量矩阵，以较少可训练参数适配模型。

## 数学形式

$$
W'=W+BA,\qquad A\in\mathbb R^{r\times d},\quad B\in\mathbb R^{k\times r},\quad r\ll\min(d,k).
$$

## 核心要点

1. $W$ 保持冻结，$A$ 和 $B$ 接受下游任务梯度。
2. 低秩参数量由 $r$ 控制，具体显存收益还取决于模型结构与实现。
3. DA-WAM 对在线视觉编码器的选定 Transformer 层使用 LoRA。

## 代表工作

- [[DA-WAM]]：LoRA 参数同时受预测和规划损失更新。

## 相关概念

- [[V-JEPA]]
