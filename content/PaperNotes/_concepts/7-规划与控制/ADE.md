---
type: concept
aliases: [Average Displacement Error, 平均位移误差]
---

# ADE

## 定义

两条离散轨迹在对应时刻的欧氏位置误差的平均值，常用于运动预测与轨迹匹配。

## 数学形式

$$
\operatorname{ADE}(\tau,\tau')=\frac{1}{H}\sum_{h=1}^{H}\|p_h-p'_h\|_2.
$$

## 核心要点

1. $H$ 是共同采样时间点数量，$p_h,p'_h$ 是对应位置。
2. ADE 衡量几何接近，不直接衡量碰撞、守法或环境响应。
3. DA-WAM 用 ADE 选出最接近专家轨迹的候选，作为未来特征监督对象。

## 代表工作

- [[DA-WAM]]：$i^{\mathrm{exp}}=\arg\min_i\operatorname{ADE}(\tau_i,\tau^{\mathrm{exp}})$。

## 相关概念

- [[轨迹评分]]
