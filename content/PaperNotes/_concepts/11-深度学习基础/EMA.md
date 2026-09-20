---
type: concept
aliases: [Exponential Moving Average, 指数移动平均, 动量编码器]
---

# EMA

## 定义

以指数衰减方式平滑跟随在线参数，常用于构建较稳定的目标网络。

## 数学形式

$$
\bar\theta\leftarrow\mu\bar\theta+(1-\mu)\theta,\qquad 0\leq\mu<1.
$$

## 核心要点

1. $\mu$ 越大，目标参数通常变化越慢。
2. 目标网络是否能防止表示坍塌，还依赖损失与架构设计。
3. DA-WAM 的 EMA 目标分支只在训练时使用。

## 代表工作

- [[DA-WAM]]：从实际未来图像提取稳定的潜变量监督目标。

## 相关概念

- [[V-JEPA]]
