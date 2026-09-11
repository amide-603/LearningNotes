# DriveWorld-VLA 论文笔记

## 1. 核心方法与创新

### 核心目标

让 **VLA 决策**与 **World Model（WM）未来预测**统一到同一个 latent space，实现：

```math
Current
\rightarrow Action
\rightarrow Future
\rightarrow Reward
\rightarrow Refine\ Action
```

即：

> **先规划动作 → 想象动作后果 → 评价未来 → 反向修正动作。**

### 三点核心创新

1. **Shared Latent Space**

   VLM 学到的隐藏状态 $H_t$ 同时提供给 **Action Head** 和 **Denoiser**，使 VLA 与 WM 在统一的 latent representation 中进行信息共享。

2. **Action-conditioned What-if Reasoning**

   将传统 World Model 的：

   ```math
   History \rightarrow Future
   ```

   提升为：

   ```math
   History + Action \rightarrow Future
   ```

   显式回答：

   > **What will happen if I take this action?**

   即模型不仅预测“未来会怎样”，还能够预测“**如果执行这个动作，未来会怎样**”。

3. **Three-stage Progressive Training**

   模型能力分三步建立：

   ```math
   Shared\ Representation
   \rightarrow
   Action\text{-}conditioned\ Future
   \rightarrow
   Future\text{-}guided\ Action\ Refinement
   ```

   即：

   - Stage 1：建立 VLA 与 WM 的共享表示；
   - Stage 2：学习动作如何影响未来世界；
   - Stage 3：利用预测出的未来反向优化动作。

---

## 2. 重要子模块

| 模块 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| VLM / Shared Latent | 多模态信息融合，建立共享 latent | 图像、BEV、历史动作、文本 | $H_t$ |
| Action Head | 预测自车未来轨迹 | $H_t,B_t,A_{t-1}$ | $A'_{future}$ |
| History-conditioned Denoiser | 根据当前/历史状态预测自然发展的未来 | $H_t,B'_t,A_{t-1}$ | Future BEV latent |
| Action-conditioned DiT | 推演执行某个动作后产生的未来 | $B'_t,A_{future},x_k,k/N$ | Flow $v_k$，迭代得到 $B'_{future}$ |
| Reward Model | 评价预测动作及其未来结果是否安全、合理 | imagined future、BEV、trajectory | Reward $\hat r$ |

### Action Head

Action Head 回答：

> **现在应该怎么开？**

```math
(H_t,B_t,A_{t-1})
\xrightarrow{Action\ Head}
A'_{future}
```

### Denoiser / World Model

Denoiser 回答：

> **未来世界会怎样？**

Stage 1：

```math
History
\rightarrow
Future
```

Stage 2：

```math
Current\ World + Action
\rightarrow
Future\ World
```

---

## 3. Action-conditioned DiT

DiT 是论文实现 **what-if imagination** 的核心模块。

其单步预测可以表示为：

```math
v_k
=
\mathrm{DiT}_{\theta}
\left(
B'_t,
A_{future},
x_k,
\frac{k}{N}
\right)
```

其中：

- $B'_t$：当前 BEV latent；
- $A_{future}$：未来动作条件；
- $x_k$：当前生成过程中的 noisy latent；
- $k/N$：当前 Flow Matching timestep；
- $v_k$：预测的 flow / denoising direction。

通过多步迭代，最终获得：

```math
B'_{future}
```

即 **action-conditioned future BEV latent**。

因此 DiT 学习的是：

```math
\boxed{
Current\ World + Action
\rightarrow
Future\ World
}
```

---

## 4. 三阶段训练

### Stage 1：VLA & World Model Joint Training

Stage 1 的目标是建立 **共享 latent space**。

输入经过 VLM 得到：

```math
H_t
=
\mathrm{VLM}
(I_t,B_t,A_{t-1},T_t)
```

$H_t$ 同时提供给：

```text
                H_t
              /     \
             ↓       ↓
      Action Head   Denoiser
             ↓       ↓
       Future Action Future BEV
```

训练目标为：

```math
\mathcal{L}_{s1}
=
\mathcal{L}_{seg}
+
\mathcal{L}_{act}
```

这一阶段的 Denoiser 主要学习：

```math
History \rightarrow Future
```

还没有显式加入未来动作条件。

---

### Stage 2：Action Controllability Fine-Tuning

Stage 2 训练 **Action-conditioned DiT**。

目标是学习：

```math
p
\left(
B_{future}
\mid
B_t,A_{future}
\right)
```

即：

> **给定当前世界和未来动作，预测这个动作会导致怎样的未来世界。**

核心能力由：

```math
History \rightarrow Future
```

升级为：

```math
History + Action \rightarrow Future
```

这就是论文中的 **action-conditioned what-if reasoning**。

---

### Stage 3：Future-Guided Evaluation & Refinement

Stage 3 建立完整决策闭环：

```math
Action
\rightarrow
Future
\rightarrow
Reward
\rightarrow
Refine\ Action
```

首先 Action Head 预测：

```math
A'_{future}
```

再将该动作送入 World Model：

```math
A'_{future}
\rightarrow
\mathrm{DiT}
\rightarrow
B'_{future}
```

Reward Model 对 imagined future 和 trajectory 进行评价，得到：

```math
\hat r
```

并利用 reward 加权 Action Loss：

```math
\mathcal{L}'_{act}
=
\hat r
\left\|
A'-A_{GT}
\right\|^2
```

从而让 Action Head 更偏向能够产生更好未来结果的动作。

---

## 5. 实验与消融结论

### 5.1 三阶段均有效

NAVSIMv1 PDMS：

| 方法 | PDMS |
| --- | ---: |
| Baseline | 87.1 |
| + Stage 1 | 87.6 |
| + Stage 2 | 89.5 |
| + Stage 3 | **91.3** |

说明：

```math
Shared\ Latent
\rightarrow
Action\text{-}conditioned\ Dynamics
\rightarrow
Future\text{-}guided\ Refinement
```

每一步均有贡献。

其中尤其重要的是：

- **Stage 2**：真正建立 $Action\rightarrow Future$；
- **Stage 3**：让预测出的 Future 反向参与 Action 优化。

因此仅仅让 WM 和 VLA **共享 latent representation 还不够**，关键还在于让 WM 建模 **动作条件下的未来动态**。

---

### 5.2 Progressive Training 很重要

如果 Stage 1 后直接联合训练 Stage 2 和 Stage 3：

```text
PDMS = 83.6
```

采用论文提出的 progressive training：

```text
PDMS = 91.3
```

提升：

```math
91.3-83.6=7.7
```

说明 **world generation 与 planning/refinement 不能简单同时优化**，需要逐阶段完成能力对齐。

因此：

> **Progressive Training 是论文最关键的训练设计之一。**

---

### 5.3 Feature-level Supervision

只使用 task-level supervision：

```text
PDMS = 87.9
```

加入 feature-level supervision：

```text
PDMS = 91.3
```

说明共享 latent 不能只依靠最终轨迹或 segmentation 等任务监督，还需要直接约束 **future latent representation**。

#### GT Future Feature 的构造

利用真实未来观测，通过冻结的 VLM：

```math
\mathcal{H}_{t+\Delta t}
=
\mathrm{VLM}_{\theta}^{*}
\left(
\mathcal{I}_{t+\Delta t},
\mathcal{B}_{t+\Delta t},
\mathcal{A}_{t+\Delta t},
\mathcal{T}_{t+\Delta t}
\right)
```

再通过冻结的 Cross-Attention：

```math
\mathcal{B}'_{t+\Delta t}
=
\mathrm{CrossAttn}_{\theta}^{*}
\left(
\mathcal{B}_{t+\Delta t},
\mathcal{H}_{t+\Delta t},
\mathcal{H}_{t+\Delta t}
\right)
```

得到：

```math
\boxed{
\mathcal{B}'_{t+\Delta t}
}
```

即 **GT future BEV latent**。

随后通过 Flow Matching Loss 监督 Action-conditioned DiT：

```math
\boxed{
\mathcal{L}_{FM}
=
\left\|
\mathrm{DiT}_{\theta}
\left(
\mathcal{B}'_{t},
\mathcal{A}_{t+\Delta t},
x_k,
\frac{k}{N}
\right)
-
\left(
\mathcal{B}'_{t+\Delta t}-x_0
\right)
\right\|_2^2
}
```

其本质是：

```math
Predicted\ Future\ Feature
\longrightarrow
GT\ Future\ Feature
```

即不仅要求模型：

> **最终轨迹预测正确**

还要求：

> **内部想象出的未来 latent 也正确。**

---

## 6. 论文最重要的启发

传统 VLA 更接近：

```math
Observation
\rightarrow
Action
```

而 DriveWorld-VLA 希望实现：

```math
\boxed{
Observation
\rightarrow
Action
\rightarrow
Imagined\ Future
\rightarrow
Evaluation
\rightarrow
Refined\ Action
}
```

因此，这篇论文真正重要的创新并不是简单的：

```math
VLA + World\ Model
```

而是让 World Model 学会：

```math
\boxed{
Action \rightarrow Future
}
```

然后再让预测出的 Future：

```math
\boxed{
Future \rightarrow Refine\ Action
}
```

最终形成：

```math
\boxed{
Action
\rightarrow
Future
\rightarrow
Reward
\rightarrow
Refine\ Action
}
```

> **核心启发：从“模仿专家怎么开”，进一步走向“先预测这样开会发生什么，再根据结果决定应该怎么开”。**