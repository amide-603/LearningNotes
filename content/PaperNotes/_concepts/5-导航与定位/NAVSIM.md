---
type: concept
aliases: [NAVSIM Benchmark, NAVSIM-v1, NAVSIM-v2]
---

# NAVSIM

## 定义

用于评估自动驾驶轨迹规划的公开基准，使用模拟的规划指标汇总候选方案的驾驶质量。

## 数学形式

$$
\mathrm{PDMS}=F(\mathrm{NC},\mathrm{DAC},\mathrm{EP},\mathrm{TTC},\mathrm{Comfort}),
$$

其中 $F$ 表示基准的指标聚合规则；此处不把它误写成各项的简单算术平均。

## 核心要点

1. v1 使用 PDMS，v2 扩展规则并报告 EPDMS。
2. 分项指标可帮助识别安全与进度之间的权衡。
3. 基准分数提升需要与真实道路闭环验证区分理解。

## 代表工作

- [[DA-WAM]]：报告 v1 navtest 93.7 PDMS、v2 navtest 87.7 EPDMS。

## 相关概念

- [[轨迹评分]]
