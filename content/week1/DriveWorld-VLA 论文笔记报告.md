# 1. 核心方法与创新

目标：让 VLA 决策与 World Model 未来预测统一到同一 latent space，实现“先规划→想象后果→评价→修正”。

核心闭环：

Current→Action→Future→Reward→RefineCurrent \rightarrow Action \rightarrow Future \rightarrow Reward \rightarrow Refine

三点创新：

1. **Shared Latent Space**：VLM 训练的隐藏层 $H_t$ 同时服务 Action Head 和 Denoiser，实现VLA、WM 隐形信息共享

2. **Action-conditioned What-if Reasoning**：从 $History\rightarrow Future$ 提升为 $History+Action\rightarrow Future$，显式推演“what happened if take this action”

3. **Three-stage Progressive Training**：先共享表示，再学习动作如何影响世界，最后用未来世界反向优化动作。

# 2. 重要子模块

|模块|作用|输入|输出|
|---|---|---|---|
|VLM / Shared Latent|多模态信息融合|图像、BEV、历史动作、文本|潜状态 $H_t$|
|Action Head|自车轨迹预测|$H_t,B_t,A_{t-1}$|$A'_{future}$|
|History Denoiser|根据历史预测自然未来|$H_t,B'_t,A_{t-1}$|latent future BEV|
|Action-conditioned DiT|逐步扩散模型，推演动作造成的未来|$B'_t,A_{future},x_k,k/N$|flow $v_k$；迭代得 $B'_{future}$|
|Reward Model|评价预测动作对应未来是否安全/合理|imagined future、BEV、trajectory|reward $\hat r$|

其中 **DiT 是核心 World Model**：

$vk=DiTθ(Bt′,Afuture,xk,k/N)v_k=DiT_\theta(B'_t,A_{future},x_k,k/N)$

通过扩散模型迭代，最终得到 action-conditioned latent future BEV

---

# 3. 三阶段任务

- **Stage 1**：共享表示 + 基础未来预测/动作预测（stage 1的Denoiser是基础未来预测，没有动作加入，stage2和3加入了动作），反向误差误差定义为 $L_{s1}=L_{seg}+L_{act}$ ，实现WM 、VLA 的耦合
    
- **Stage 2**：训练 Action-conditioned DiT，学习动作如何影响世界，即 $p(B_{future}|B_t,A_{future})$
    
- **Stage 3**：$Action\rightarrow Future\rightarrow Reward$，利用 $L'_{act}=\hat r|A'-A_{GT}|^2$ 修正 Action Head。
    

---

# 4. 实验结论

1. NAVSIM PDMS：Baseline 87.1 → Stage1 87.6 → Stage2 89.5 → Stage3 **91.3**；说明 WM 和VLA 仅仅共享潜状态还不够，真正缺少的是真实世界中的变化推进，**Stage2 的动作条件想象和 Stage3 的未来反馈最关键**

2. Stage2+3若直接联合训练仅 **83.6**，渐进训练达到 **91.3**，说明 执行动作后的世界和动作优化不能同时训练，**progressive training 是最关键设计之一**。

3. 加入 feature-level supervision 后 PDMS 从 **87.9→91.3**，说明共享 latent 不能只靠最终任务监督，需要显式特征对齐

    特征监督：利用真实未来观测经过冻结的 VLM 和 Cross-Attention 得到 GT future BEV latent $\mathcal{B}'_{t+\Delta t}$，再通过 Flow Matching loss $\mathcal{L}_{FM}$ 监督 action-conditioned DiT，使模型学习动作条件下的未来特征演化
![[Pasted image 20260911182529.png]]