---
type: concept
aliases: [Video Joint Embedding Predictive Architecture, 视频联合嵌入预测架构]
---

# V-JEPA

## 定义

视频联合嵌入预测架构，在表征空间预测被遮蔽或未来视频内容的特征，而非逐像素重建画面。

## 数学形式

$$
\mathcal L_{\mathrm{JEPA}}=\ell\bigl(P_\phi(E_\theta(X_t),a),\operatorname{sg}(E_{\bar\theta}(X_{t+\Delta}))\bigr).
$$

这是用于理解 DA-WAM 的概括式，具体 V-JEPA 预训练任务可能采用不同条件与损失。

## 核心要点

1. 潜变量预测可避免像素生成中的部分细节开销。
2. 在线编码器与目标编码器的更新方式会影响目标稳定性。
3. DA-WAM 使用预训练 V-JEPA 2.1，并在规划训练中适配其在线分支。

## 代表工作

- [[DA-WAM]]：用 V-JEPA 2.1 的密集潜变量目标监督专家匹配轨迹。

## 相关概念

- [[EMA]]
- [[LoRA]]
