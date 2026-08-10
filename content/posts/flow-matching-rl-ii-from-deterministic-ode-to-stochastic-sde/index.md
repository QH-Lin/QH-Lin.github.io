---
title: "流匹配的强化学习（二）：从确定性 ODE 到随机 SDE"
date: 2026-08-10T12:05:39+08:00
draft: false
tags:
  - flow matching
  - reinforcement learning
  - stochastic differential equations
categories:
  - AI
description: "从确定性概率流到具有随机转移、探索能力与可计算路径似然的生成策略。"
showToc: true
TocOpen: false
math: true
---

## 从“确定性概率流”到“可探索、可计算似然的生成策略”

本文希望回答一个核心问题：

> 一个已经用 Flow Matching 学到的确定性 ODE 生成器，怎样使用奖励继续优化？为什么 Flow-GRPO 一类方法要把 ODE 改写成 SDE？这种改写与“ODE、SDE 可以具有相同单时刻边际分布”的统一框架有什么关系？

全文最重要的逻辑链是

$$
\boxed{
\begin{array}{c}
\text{Flow Matching 先学习速度场 }u_\theta(t,x)
\\[1mm]
\Downarrow
\\[1mm]
\text{ODE 将噪声输运到生成分布}
\\[1mm]
\Downarrow
\\[1mm]
\text{奖励给出“哪些生成结果更好”}
\\[1mm]
\Downarrow
\\[1mm]
\text{若采用逐步 PPO/GRPO 策略梯度，}
\\
\text{就需要随机转移、探索和可计算的转移密度}
\\[1mm]
\Downarrow
\\[1mm]
\text{用 score 修正漂移，将 ODE 改写成同边际 SDE}
\\[1mm]
\Downarrow
\\[1mm]
\text{离散 SDE 给出 Gaussian 转移概率}
\\[1mm]
\Downarrow
\\[1mm]
\text{用奖励优势更新速度场。}
\end{array}}
$$

不过，必须立刻加上一个范围限定：

$$
\boxed{
\text{Flow Matching 做强化学习}
\;\not\Rightarrow\;
\text{数学上必须把 ODE 变成 SDE。}
}
$$

更准确的说法是：

$$
\boxed{
\begin{array}{c}
\text{若把每个去噪步当作随机策略动作，}
\\
\text{并使用 PPO/GRPO 的逐步似然比，}
\\
\text{那么通常需要把确定性 ODE 随机化；}
\\
\text{同边际 SDE 是最自然、最有理论依据的一条路线。}
\end{array}}
$$

奖励加权回归、Flow Matching 代理似然、可微奖励反向传播、Q 引导、离线 RL、Adjoint Matching 等路线，并不都要求 SDE。本文会先完整解释 ODE 到 SDE 的主线，再说明这些例外为什么不矛盾。

---

## 1. 本文从第一篇的哪一步开始

本文是[《流匹配的 RL（一）：统一理解 Diffusion 与 Flow Matching》](/posts/flow-matching-rl-from-unified-diffusion-and-flow-matching/)的续篇。第一篇已经完整讨论：

- 随机插值怎样定义概率路径；
- 条件平均速度为什么满足连续性方程；
- Flow Matching 的平方回归为什么学到速度场；
- ODE 与带 score 补偿的 SDE 为什么可以具有相同单时刻边际；
- Rectified Flow、Diffusion、DDPM 与 DDIM 怎样落入同一框架。

因此本文不再重复这些证明，只把第一篇的结论当作接口。

### 1.1 本文直接使用的三个结论

第一，当前 Flow Matching 模型给出速度场。第一篇主要写作 $b_\theta$；为了贴近 Flow-GRPO 等 RL 论文，本文记作

$$
u_\theta(t,x,c)\equiv b_\theta(t,x,c).
\tag{1.1}
$$

第二，从基分布出发运行 ODE

$$
\frac{\mathrm dX_t}{\mathrm dt}
\mathrel{=}
u_\theta(t,X_t,c)
\tag{1.2}
$$

会定义当前模型的中间边际 $\rho_\theta(t)$ 和终点生成分布 $p_\theta(\cdot\mid c)$。

第三，在理想连续时间下，若

$$
s_\theta(t,x,c)
\mathrel{=}
\nabla_x\log\rho_\theta(t,x\mid c),
\tag{1.3}
$$

则可以选择适当的 score 补偿漂移，为同一组边际构造随机 SDE。

本文真正新增的问题是：

> 为什么生成模型一旦要接入逐步 PPO/GRPO，确定性 ODE 的路径结构就不够用了？同边际 SDE 又怎样把它变成可探索、可计算转移概率的策略？

### 1.2 本文新增的三类时间

第一篇主要研究连续生成时间。RL 场景还要额外区分离散去噪步与环境时间：

| 符号 | 含义 |
|---|---|
| $t\in[0,1]$ | Flow Matching 内部的连续生成时间 |
| $k=0,\ldots,K$ | ODE/SDE 数值离散后的内部去噪步 |
| $h=0,1,\ldots$ | 机器人或一般 RL 环境的外层决策时间 |

图像生成通常在一条内部轨迹结束后得到终点奖励。机器人策略则在每个环境时刻 $h$ 内部运行 $K$ 个去噪步，生成动作块 $a_h$，再由环境给出奖励。

所以后文必须区分：

$$
\boxed{
\text{内部去噪 MDP}
\neq
\text{外层环境 MDP}.
}
$$

---

## 2. 从生成分布切换到奖励优化

给定条件 $c$ 和当前参数 $\theta$，本文把 Flow Matching 模型视为一个能够采样

$$
x\sim p_\theta(\cdot\mid c)
\tag{2.1}
$$

的策略。$c$ 在图像生成中可以是文本提示，在机器人中可以包含观测、语言指令和历史状态。

预训练已经解决“怎样从噪声得到符合数据分布的样本”。第二篇从这里开始，只增加三个 RL 对象：

| 对象 | 记号 | 含义 |
|---|---|---|
| 当前策略 | $p_\theta(x\mid c)$ | 当前 ODE 或其他 sampler 的终点分布 |
| 参考策略 | $p_{\mathrm{ref}}(x\mid c)$ | RL 前的预训练模型 |
| 奖励 | $R(x,c)$ | 衡量终点是否满足任务目标 |

接下来的问题不再是重新证明概率路径，而是决定：

1. 奖励希望把终点概率质量移到哪里；
2. 用终点似然、路径似然还是回归代理实现这种移动；
3. 是否需要把确定性内部动力学改成随机策略。

---

## 3. 强化学习到底要优化什么

### 3.1 从数据分布变成奖励倾斜分布

设预训练生成模型的终点分布为

$$
p_{\mathrm{ref}}(x\mid c),
$$

其中 $c$ 可以是文本提示、机器人观测或其他条件。

给定终点奖励 $R(x,c)$，一种标准的 KL 正则化目标是

$$
\boxed{
\max_p
\left\{
\mathbb E_{x\sim p(\cdot\mid c)}[R(x,c)]
\mathbin{-}
\beta
D_{\mathrm{KL}}
\bigl(
p(\cdot\mid c)
\|
p_{\mathrm{ref}}(\cdot\mid c)
\bigr)
\right\}.
}
\tag{3.1}
$$

对分布 $p$ 做变分优化，最优终点分布满足

$$
\boxed{
p^*(x\mid c)
\mathrel{=}
\frac{1}{Z(c)}
p_{\mathrm{ref}}(x\mid c)
\exp\!\left(\frac{R(x,c)}{\beta}\right).
}
\tag{3.2}
$$

其中

$$
Z(c)
\mathrel{=}
\int
p_{\mathrm{ref}}(x\mid c)
\exp\!\left(\frac{R(x,c)}{\beta}\right)
\mathrm dx
$$

是归一化常数。

式 (3.2) 给出了 Flow Matching RL 最清楚的分布层解释：

$$
\boxed{
\text{RL 不是凭空创造另一类生成模型，}
\quad
\text{而是把预训练终点分布向高奖励区域重新加权。}
}
$$

### 3.2 奖励改变的是终点质量偏好

预训练 Flow Matching 最小化速度回归误差，其目标是复现训练数据分布。

RL 后训练关心的是

$$
\text{生成结果是否满足组合、文字、审美、安全或任务成功等要求。}
$$

这些要求常常：

- 没有逐像素监督标签；
- 不可微；
- 只能在最终样本上评估；
- 可能来自规则、检测器、VLM、人工偏好模型或真实环境。

因此，Flow Matching 的监督学习标签

$$
x_1-x_0
$$

与 RL 的奖励

$$
R(x_1,c)
$$

承担完全不同的角色。

### 3.3 RL 后概率路径也会改变

若终点从 $p_{\mathrm{ref}}$ 变成 $p^*$，相应的中间边际、速度场和 score 通常也会改变：

$$
\rho_{\mathrm{ref}}(t)
\longrightarrow
\rho^*(t),
$$

$$
u_{\mathrm{ref}}(t,x)
\longrightarrow
u^*(t,x),
$$

$$
s_{\mathrm{ref}}(t,x)
\longrightarrow
s^*(t,x).
$$

所以“ODE 与 SDE 同边际”应当在固定参数或固定目标路径下理解。

在某一次 rollout 中，可以让当前模型的 ODE 与随机化 SDE 具有相同边际。RL 更新参数以后，当前模型本身变了，新的概率路径也随之改变。

---

## 4. 为什么确定性 ODE 不能直接套用逐步 PPO/GRPO

### 4.1 ODE 仍然可以生成不同样本

先澄清一个常见误解。

ODE 动力学是确定性的，但生成模型并不是只能生成一个样本。因为初始条件仍然随机：

$$
X_0\sim\mathcal N(0,\mathrm{Id}).
$$

不同的初始噪声会产生不同的 ODE 轨迹和不同终点。

因此

$$
\boxed{
\text{ODE 确定性}
\neq
\text{整个生成分布没有随机性。}
}
$$

真正的问题在于：给定某个内部状态后，下一步转移是确定的。

### 4.2 ODE 离散步是 Dirac 转移

用 Euler 法离散式 (1.2)：

$$
X_{k+1}
\mathrel{=}
X_k+h_k u_\theta(t_k,X_k).
\tag{4.1}
$$

令

$$
F_\theta(X_k)
\mathrel{=}
X_k+h_k u_\theta(t_k,X_k).
$$

则条件转移为

$$
\boxed{
\pi_\theta(X_{k+1}\mid X_k)
\mathrel{=}
\delta_{F_\theta(X_k)}(X_{k+1}).
}
\tag{4.2}
$$

这里 $\delta$ 是 Dirac 测度。

它不是通常意义下相对于 Lebesgue 测度的光滑概率密度。

### 4.3 PPO/GRPO 需要似然比

PPO 的核心量是

$$
r_k(\theta)
\mathrel{=}
\frac{
\pi_\theta(X_{k+1}\mid X_k,c)
}{
\pi_{\theta_{\mathrm{old}}}(X_{k+1}\mid X_k,c)
}.
\tag{4.3}
$$

GRPO 仍然使用类似的策略比，只是用同一条件下的一组奖励构造相对优势，而不一定训练单独的 value 网络。

如果新旧策略都是 Dirac 转移，

$$
\pi_\theta
\mathrel{=}
\delta_{F_\theta(X_k)},
\qquad
\pi_{\theta_{\mathrm{old}}}
\mathrel{=}
\delta_{F_{\theta_{\mathrm{old}}}(X_k)},
$$

那么只要两个映射的输出不同，它们的支撑就可能不重合。此时普通密度比不能像 Gaussian 策略那样稳定计算，路径 KL 也可能退化。

### 4.4 直接计算 ODE 终点似然很贵

连续归一化流可以用瞬时变量替换公式计算终点密度：

$$
\boxed{
\log q_\theta(X_1)
\mathrel{=}
\log q_0(X_0)
\mathbin{-}
\int_0^1
\nabla\cdot
u_\theta(t,X_t)
\mathrm dt.
}
\tag{4.4}
$$

但高维模型中的散度

$$
\nabla\cdot u_\theta
\mathrel{=}
\operatorname{tr}
\left(
\frac{\partial u_\theta}{\partial x}
\right)
$$

计算昂贵，通常需要 Hutchinson trace estimator、额外的向量—Jacobian 乘积和 ODE 积分。

这使得在大量在线样本、多个 PPO epoch 和数十个去噪步中反复计算 ODE 似然非常昂贵。

### 4.5 逐步探索也受到限制

确定性 ODE 的随机性只来自初始噪声。一旦给定内部状态 $X_k$，后续分支唯一。

而逐步策略优化希望具有

$$
X_{k+1}
\sim
\pi_\theta(\cdot\mid X_k,c),
$$

从同一状态附近探索多个可能方向。

所以 Flow-GRPO 需要的不是抽象意义上的“任何随机性”，而是：

$$
\boxed{
\text{条件于当前去噪状态的局部随机转移。}
}
$$

---

## 5. 从 ODE 构造同边际 SDE

第一篇第 10 节已经完整证明了概率流 ODE、Fokker--Planck 方程与同边际 SDE 的关系。本节不重复该推导，只提取 RL 后训练所需的接口。

设当前 Flow Matching 模型的 ODE 为

$$
\frac{\mathrm dX_t}{\mathrm dt}
\mathrel{=}
u_\theta(t,X_t).
$$

其边际密度 $\rho_\theta$ 满足

$$
\partial_t\rho_\theta
\mathrel{=}
-\nabla\cdot(u_\theta\rho_\theta),
\tag{5.1}
$$

记当前边际的 score 为

$$
s_\theta(t,x)
:=
\nabla_x\log\rho_\theta(t,x).
\tag{5.2}
$$

对于任意非负噪声日程 $\kappa(t)$，可构造

$$
\boxed{
\mathrm dZ_t
\mathrel{=}
\bigl(
u_\theta+\kappa s_\theta
\bigr)(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\kappa(t)}\,\mathrm dW_t.
}
\tag{5.3}
$$

这里不能省略 $\kappa s_\theta$ 而只给 ODE 加噪声；它负责抵消扩散对密度方程造成的改变。代入 Fokker--Planck 方程后，核心只剩一行：

$$
-\kappa\nabla\cdot(\rho_\theta s_\theta)
\mathbin{+}
\kappa\Delta\rho_\theta
=0.
\tag{5.4}
$$

因此，在相同初始分布以及适当正则性与唯一性条件下，

$$
\boxed{
\mathcal L(Z_t)=\mathcal L(X_t)=\rho_\theta(t),
\qquad 0\le t\le1.
}
\tag{5.5}
$$

这条结论只保证**相同单时刻边际**，不保证相同的：

- 单条轨迹；
- 相邻时刻联合分布；
- 条件转移概率；
- 路径熵与路径似然。

而这些“不相同”正是 RL 所利用的自由度：终点生成分布在连续时间极限下保持一致，内部路径却获得了随机分支、探索以及可计算的条件转移。

$$
\boxed{
\text{保持边际}
\quad+\quad
\text{重写路径统计}
\quad\Longrightarrow\quad
\text{可用于逐步策略梯度的生成策略。}
$$

> 完整证明与更一般的漂移自由度见[第一篇第 10 节](/posts/flow-matching-rl-from-unified-diffusion-and-flow-matching/#10-同一条边际路径的-sde)。本文接下来只讨论这项自由度怎样变成 RL 的策略接口。

---

## 6. 为什么同边际 SDE 可以变成 RL 策略

### 6.1 Euler--Maruyama 离散

将式 (5.3) 离散化：

$$
\boxed{
Z_{k+1}
\mathrel{=}
Z_k
\mathbin{+}
h_k
\bigl(
u_\theta+\kappa_k s_\theta
\bigr)(t_k,Z_k)
\mathbin{+}
\sqrt{2\kappa_k h_k}\,\epsilon_k,
}
\tag{6.1}
$$

其中

$$
\epsilon_k\sim\mathcal N(0,\mathrm{Id}).
$$

给定 $Z_k=z$，下一步服从 Gaussian：

$$
\boxed{
\pi_\theta(Z_{k+1}\mid Z_k=z,c)
\mathrel{=}
\mathcal N
\left(
\mu_\theta(t_k,z,c),
2\kappa_k h_k\,\mathrm{Id}
\right),
}
\tag{6.2}
$$

其中

$$
\mu_\theta
\mathrel{=}
z
\mathbin{+}
h_k
\bigl(
u_\theta+\kappa_k s_\theta
\bigr)(t_k,z,c).
\tag{6.3}
$$

### 6.2 转移对数概率变得可计算

Gaussian 转移的对数概率为

$$
\begin{aligned}
\log\pi_\theta(z_{k+1}\mid z_k,c)
=&
-\frac{d}{2}
\log(4\pi\kappa_k h_k)
\\
&
-\frac{
\|z_{k+1}-\mu_\theta(t_k,z_k,c)\|^2
}{
4\kappa_k h_k
}.
\end{aligned}
\tag{6.4}
$$

因此 PPO/GRPO 所需的似然比

$$
r_k(\theta)
\mathrel{=}
\exp
\left[
\log\pi_\theta(z_{k+1}\mid z_k,c)
\mathbin{-}
\log\pi_{\theta_{\mathrm{old}}}(z_{k+1}\mid z_k,c)
\right]
\tag{6.5}
$$

可以直接计算。

### 6.3 KL 正则也变得简单

若当前策略与参考策略使用相同协方差，则

$$
\boxed{
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid z_k,c)
\|
\pi_{\mathrm{ref}}(\cdot\mid z_k,c)
\right)
\mathrel{=}
\frac{
\|\mu_\theta-\mu_{\mathrm{ref}}\|^2
}{
4\kappa_k h_k
}.
}
\tag{6.6}
$$

这避免了连续 ODE 散度积分。

### 6.4 去噪过程成为内部 MDP

可以把内部生成过程写成

| RL 对象 | 去噪过程中的对应物 |
|---|---|
| 状态 $s_k$ | 当前 latent、时间、条件：$(Z_k,t_k,c)$ |
| 动作 $a_k$ | 采样得到的下一 latent $Z_{k+1}$ |
| 策略 | Gaussian 转移 $\pi_\theta(Z_{k+1}\mid Z_k,c)$ |
| 轨迹 | $(Z_0,Z_1,\ldots,Z_K)$ |
| 奖励 | 通常只在终点 $Z_K$ 解码后给出 |

这不是说 latent 在语义上真的是机器人动作，而是说它在策略梯度的数学形式中扮演“动作”的角色。

---

## 7. Flow-GRPO 的完整逻辑

### 7.1 同一条件下生成一组样本

给定条件 $c$，从当前随机化 flow 策略生成 $G$ 条轨迹：

$$
\tau_i
\mathrel{=}
\bigl(
Z_0^i,Z_1^i,\ldots,Z_K^i
\bigr),
\qquad
i=1,\ldots,G.
$$

终点解码得到样本 $x^i$，计算奖励

$$
R_i=R(x^i,c).
$$

### 7.2 组相对优势

Flow-GRPO 使用组内标准化：

$$
\boxed{
\widehat A_i
\mathrel{=}
\frac{
R_i-\operatorname{mean}(R_1,\ldots,R_G)
}{
\operatorname{std}(R_1,\ldots,R_G)+\varepsilon
}.
}
\tag{7.1}
$$

这一步不需要单独训练 value 网络。

由于奖励通常只在终点给出，最简单的做法是把同一个终点优势分配给轨迹中的所有内部步：

$$
\widehat A_{i,k}
\mathrel{=}
\widehat A_i.
\tag{7.2}
$$

这也是一个明显的近似：不同去噪步对最终奖励的贡献并不相同。

### 7.3 Clipped 策略目标

定义

$$
r_{i,k}(\theta)
\mathrel{=}
\frac{
\pi_\theta(Z_{k+1}^i\mid Z_k^i,c)
}{
\pi_{\theta_{\mathrm{old}}}(Z_{k+1}^i\mid Z_k^i,c)
}.
\tag{7.3}
$$

Flow-GRPO 的核心目标可写成

$$
\boxed{
\begin{aligned}
\mathcal J_{\mathrm{GRPO}}(\theta)
\mathrel{=}
\mathbb E
\Bigg[
\frac{1}{GK}
\sum_{i=1}^G
\sum_{k=0}^{K-1}
&
\min
\Bigl(
r_{i,k}(\theta)\widehat A_i,
\\
&
\operatorname{clip}
(r_{i,k}(\theta),1-\epsilon,1+\epsilon)
\widehat A_i
\Bigr)
\\
&
-\beta
D_{\mathrm{KL}}
(\pi_\theta\|\pi_{\mathrm{ref}})
\Bigg].
\end{aligned}
}
\tag{7.4}
$$

### 7.4 为什么 ODE 到 SDE 是这条路线的关键

Flow-GRPO 同时需要：

1. 同一提示下的多样 rollout；
2. 给定当前 latent 后仍可继续随机探索；
3. 每一步的转移密度；
4. 新旧策略的似然比；
5. 当前策略与参考策略的 KL。

同边际 SDE 一次性提供了这些结构。

因此，对 Flow-GRPO 而言，可以写成

$$
\boxed{
\text{ODE}
\xrightarrow[\text{score 修正漂移}]{\text{加入扩散}}
\text{同边际 SDE}
\xrightarrow{\text{离散}}
\text{Gaussian 去噪策略}
\xrightarrow{\text{GRPO}}
\text{奖励优化。}
}
$$

---

## 8. Rectified Flow 中怎样从速度得到 score

同边际 SDE 需要 $s_\theta$，而 Flow Matching 模型通常只输出速度。这个参数化转换已在第一篇第 13、16 节讨论；此处只记录实现 Flow-GRPO 所需的最终公式。

对线性路径

$$
x_t=(1-t)x_0+tx_1,
\qquad
x_0\sim\mathcal N(0,\mathrm{Id}),
\tag{8.1}
$$

若 $u(t,x)$ 是对应的理想速度场，则边际 score 可写为

$$
\boxed{
s(t,x)
\mathrel{=}
\frac{
t\,u(t,x)-x
}{
1-t
}.
}
\tag{8.2}
$$

因此在理想线性路径下，可直接由 velocity 构造式 (5.3) 的 score 补偿项，不一定另训 score 网络。有限模型误差和端点附近的数值放大会影响实际实现，不能把这个恒等式理解为无误差转换。

> **时间方向备注。**
>
> 本文沿用“$t=0$ 是噪声，$t=1$ 是数据”的约定。Flow-GRPO、部分扩散论文和代码常用相反方向：$t=1$ 是噪声，采样时积分到 $t=0$。令 $\tau=1-t$ 后，漂移中的正负号和分母形式会改变。比较公式时必须先统一时间方向，不能只看表面上的加号或减号。
>
> 完整的 velocity、noise 与 score 推导见[第一篇第 13、16 节](/posts/flow-matching-rl-from-unified-diffusion-and-flow-matching/#13-linear-flow-matching-与-rectified-flow)。

---

## 9. “ODE 与 SDE 同边际”在 RL 中究竟保证了什么

### 9.1 固定当前模型时的保证

设某次 rollout 前参数固定为 $\theta_{\mathrm{old}}$。

理想连续时间下，若 score 精确，则：

$$
\boxed{
\operatorname{Law}(X_t)
\mathrel{=}
\operatorname{Law}(Z_t)
\quad
\text{对所有 }t.
}
\tag{9.1}
$$

特别地，终点分布相同：

$$
\operatorname{Law}(X_1)
\mathrel{=}
\operatorname{Law}(Z_1).
\tag{9.2}
$$

这意味着可以在不故意改变当前生成器单时刻分布的前提下，为训练 rollout 增加路径随机性。

### 9.2 它不保证有限步离散仍然精确

实际训练使用有限步数 $K$，并采用近似速度、近似 score 和数值离散。

因此实际链满足的通常只是

$$
p_{\theta,K}(t_k)
\approx
\rho_\theta(t_k),
$$

而不是严格相等。

误差来源包括：

- $u_\theta$ 不是精确速度；
- 由 $u_\theta$ 换算的 score 不精确；
- Euler--Maruyama 是一阶离散；
- 训练常把步数降到 4、6、10 或 20；
- classifier-free guidance 会改变有效速度场；
- 噪声日程在端点附近可能数值不稳定。

### 9.3 SDE sampler 是策略的一部分

在普通生成推理中，采样器常被看成模型之外的数值工具。

但在逐步策略梯度中，转移概率直接进入损失：

$$
\log\pi_\theta(Z_{k+1}\mid Z_k,c).
$$

所以噪声大小、离散步长和转移公式都会改变训练策略本身。

这意味着：

$$
\boxed{
\text{RL 中的 sampler 设计}
\neq
\text{无关紧要的工程细节。}
}
$$

---

## 10. SDE 路线的主要技术问题

### 10.1 探索与稳定性的矛盾

扩散率 $\kappa(t)$ 太小：

- 组内样本差异小；
- 奖励标准差小；
- 优势估计不稳定或接近零；
- 探索高奖励区域很慢。

扩散率 $\kappa(t)$ 太大：

- latent 偏离预训练路径；
- 模型在分布外状态上预测；
- 终点出现噪声伪影；
- 奖励模型的排序可能失真；
- 策略梯度方差增大。

所以不能简单地认为“噪声越大，RL 越好”。

### 10.2 连续时间正确不等于少步离散正确

连续时间 SDE 的边际等价依赖 Fokker--Planck 方程。

但 Euler--Maruyama 的一步更新冻结了步首的速度和 score。新注入的噪声会立即改变 latent，而步内漂移没有及时响应这个变化。

在步长很小时，这是一阶可接受近似；在少步训练时，误差可能很大。

### 10.3 Flow-GRPO 后续工作的争论

Coefficients-Preserving Sampling 指出，Flow-GRPO 式的 Euler SDE 在少步采样时可能注入过量噪声，并提出类似 DDIM 的系数保持随机转移。

PRECISE 随后进一步指出：

1. Euler 式转移可能产生过量离散噪声；
2. 只保持信号和噪声系数仍不保证一般数据分布的正确边际；
3. 需要同时考虑探索日程与 SDE 一致的有限步转移。

PRECISE 在一步内冻结 clean latent 的后验均值，从而得到局部线性 SDE 的闭式 Gaussian 转移。

这些工作共同说明：

$$
\boxed{
\text{ODE 到 SDE 的连续理论只是起点，}
\quad
\text{少步随机采样器仍需要单独设计和验证。}
}
$$

### 10.4 终点奖励造成粗糙 credit assignment

若所有内部步共享同一个终点优势，

$$
\widehat A_{i,k}=\widehat A_i,
$$

就隐含假设每一步对终点奖励承担相同责任。

但图像生成中常见的经验是：

- 早期步更影响全局布局与对象数量；
- 中期步更影响形状和语义；
- 后期步更影响纹理、文字与细节。

因此，step-aware advantage、过程奖励和密集奖励成为后续研究方向。

---

## 11. ODE 变 SDE 并不是所有 RL 的必要条件

这一章给出全文最重要的边界。

### 11.1 必须先问：使用哪一种 RL 更新

“Flow Matching 能不能做 RL”不是一个单一算法问题。

需要先问：

1. 奖励是否可微？
2. 是否需要严格的 on-policy 策略梯度？
3. 是否使用逐步 PPO/GRPO 似然比？
4. 是否能接受代理似然而非真实似然？
5. 是否有离线高奖励数据？
6. 是否训练模型，还是只做推理时搜索？

不同答案会产生不同路线。

### 11.2 路线 A：逐步策略似然比

代表：

- Flow-GRPO；
- DanceGRPO；
- 一部分把去噪链视作 MDP 的 PPO 方法；
- ReinFlow 的离散 Markov 噪声注入路线；
- $\pi_{\mathrm{RL}}$ 的 Flow-Noise 与 Flow-SDE 路线。

这类方法需要随机条件转移和可计算的逐步概率。

其中 Flow-GRPO 使用同边际 SDE；ReinFlow 则注入可学习噪声，把离散 flow 变成 Gaussian Markov 过程。后者的目的也是得到探索和精确的离散路径似然，但不等同于“严格保持原 ODE 所有连续时间边际”。

$\pi_{\mathrm{RL}}$ 把这两种思想同时用于 flow-based VLA：

1. Flow-Noise 使用可学习噪声网络，把动作去噪过程写成离散 Markov 过程；
2. Flow-SDE 使用 ODE-to-SDE 转换保持理想连续时间边际；
3. 再把内部去噪 MDP 嵌入机器人与环境交互的外层 MDP。

对这一路线，可以近似地说：

$$
\boxed{
\text{需要把确定性内部过程随机化。}
}
$$

### 11.3 路线 B：奖励加权 Flow Matching

考虑非负权重

$$
w(x,c)\ge0.
$$

训练

$$
\boxed{
\mathcal L_{\mathrm{RWFM}}(\theta)
\mathrel{=}
\mathbb E
\left[
w(x_1,c)
\left\|
u_\theta(t,x_t,c)-Y_t
\right\|^2
\right].
}
\tag{11.1}
$$

在理想条件下，权重会把目标分布变成

$$
p_w(x\mid c)
\propto
w(x,c)p_{\mathrm{data}}(x\mid c).
\tag{11.2}
$$

若取

$$
w(x,c)
\mathrel{=}
\exp\!\left(\frac{R(x,c)}{\beta}\right),
\tag{11.3}
$$

就与式 (3.2) 的奖励倾斜分布一致。

Online Reward-Weighted CFM、RWFM 等方法可以从当前模型采样终点、计算奖励，再做奖励加权速度回归。它们不需要逐步 ODE 转移密度，也不必把 rollout 改成 SDE。

代价是：

- 它更像奖励加权回归或交叉熵方法；
- 容易向少数高奖励模式坍缩；
- 需要正则化、数据混合或 Wasserstein 约束；
- 严格的 trust-region 策略梯度解释较弱。

### 11.4 路线 C：用 Flow Matching 损失代理似然

Flow Policy Optimization 提出，用条件 Flow Matching 损失近似 ELBO，再构造 PPO 风格的代理比：

$$
\boxed{
\widehat r_{\mathrm{FPO}}(\theta)
\mathrel{=}
\exp
\left(
\widehat{\mathcal L}_{\mathrm{CFM},\theta_{\mathrm{old}}}
\mathbin{-}
\widehat{\mathcal L}_{\mathrm{CFM},\theta}
\right).
}
\tag{11.4}
$$

它把真实策略似然比替换成 Flow Matching 代理比。

这样 rollout 可以使用：

- 确定性 ODE；
- 随机 SDE；
- 高阶 solver；
- 不同的采样步数。

所以 FPO 是“on-policy、PPO 风格，但不要求 ODE→SDE”的直接反例。

需要注意：它优化的是代理比，不是精确终点密度比；小 Monte Carlo 样本下还会出现比值估计偏差。

### 11.5 路线 D：终点采样加解析加噪回归

Reinforce Adjoint Matching 从 KL 正则化最优控制和 REINFORCE 恒等式出发，构造奖励修正的回归目标。

它的训练结构是：

$$
\boxed{
\begin{array}{c}
\text{用任意 sampler 生成当前模型终点}
\\
\Downarrow
\\
\text{计算终点奖励}
\\
\Downarrow
\\
\text{像预训练一样解析地给终点加噪}
\\
\Downarrow
\\
\text{回归奖励修正后的速度目标。}
\end{array}}
$$

RAM 明确不需要：

- SDE rollout；
- reward gradient；
- 沿 SDE 的反向 adjoint sweep。

它仍然用到了与 SDE 最优控制有关的理论，但实际训练 rollout 可以使用 ODE。这说明“理论中出现 SDE”与“算法必须用 SDE 采样”是两件事。

### 11.6 路线 E：可微奖励或环境模型

若奖励 $R(x)$ 可微，可以直接优化

$$
\mathcal J(\theta)
\mathrel{=}
\mathbb E_{x_0}
\left[
R(X_1^\theta)
\right],
\tag{11.5}
$$

并通过 ODE solver 或其 adjoint 对 $\theta$ 求梯度。

这条路线不需要策略似然比，也不必随机化每个内部步。

缺点是：

- 需要可微奖励；
- 要反向传播穿过整个采样轨迹；
- 内存或 adjoint 数值误差可能很大；
- 黑盒人类偏好与真实环境奖励通常不可微。

### 11.7 路线 F：离线 RL、Q 引导与蒸馏

FlowQ、Energy-Weighted Flow Matching、Q-guided flow policy 等方法利用离线数据和 critic，把高价值动作或能量信息转化为：

- 权重；
- guidance；
- 新的速度标签；
- 蒸馏目标；
- density transport。

它们不一定需要 on-policy SDE 路径似然。

### 11.8 路线 G：把 RL 经验蒸馏回条件式 Flow Matching

$\pi_{0.7}$ 展示了另一种与 RL 连接、但本身不执行在线策略梯度的路线。

它的 flow matching action expert 接收的不只是任务指令，还接收包含行为质量与策略信息的上下文：

$$
\boxed{
C_h
\mathrel{=}
\bigl(
\text{任务与子任务语言},
\text{subgoal 图像},
\text{速度},
\text{质量},
\text{错误标记},
\text{控制模式}
\bigr).
}
\tag{11.6}
$$

训练数据不仅包括人工示范，还包括：

- 旧策略的自主 rollout；
- 失败或次优轨迹；
- RL 后训练 specialist 产生的高质量经验；
- 人类干预与其他机器人、非机器人数据。

若不给出质量上下文，混合质量数据可能使模型平均不同策略并学到次优行为。$\pi_{0.7}$ 用 episode metadata 把这些模式显式分开；推理时再提示高质量、无错误和期望速度。

从分布角度，可以把它理解成学习

$$
p_\theta(a_{h:h+H}\mid o_h,C_h),
\tag{11.7}
$$

再通过选择 $C_h$ 控制希望采样的条件切片。

它与直接 RL 的区别是：

$$
\boxed{
\begin{array}{c}
\text{RL specialist 先产生高价值经验，}
\\
\text{通用 flow policy 再用条件式监督学习吸收这些经验。}
\end{array}}
$$

所以 $\pi_{0.7}$ 更接近 RL 经验蒸馏、质量条件建模与 steerable behavior cloning，而不是 ODE-to-SDE 的在线 RL 算法。

### 11.9 更准确的结论

| 方法类型 | 是否通常需要 ODE→SDE | 原因 |
|---|---:|---|
| Flow-GRPO 式逐步 PPO/GRPO | 是 | 需要局部探索和 Gaussian 转移似然 |
| ReinFlow 式 Markov 噪声注入 | 需要随机化，但不一定是同边际连续 SDE | 需要离散路径似然与探索 |
| $\pi_{\mathrm{RL}}$ | Flow-SDE 分支需要；Flow-Noise 分支采用离散随机化 | 分别构造两层 MDP 或可计算的离散路径似然 |
| 奖励加权 Flow Matching | 否 | 直接做加权速度回归 |
| FPO | 否 | 用 CFM/ELBO 代理似然比 |
| RAM | 否 | 终点采样后解析加噪并回归 |
| 可微奖励路径反传 | 否 | 直接重参数化求梯度 |
| 离线 Q 引导或蒸馏 | 否 | 不依赖 on-policy 逐步概率 |
| $\pi_{0.7}$ 式 RL 经验蒸馏 | 否 | 用质量上下文对 RL、自主和失败数据做条件式 Flow Matching |
| SMC、树搜索等推理时对齐 | 需要随机转移，但不必限定为 SDE | GLASS Flows 等可用 ODE 采样随机转移 |

---

## 12. 第一篇与第二篇的分工

两篇文章共享同一套数学对象，但回答的问题不同：

| 第一篇：统一理解生成动力学 | 第二篇：把生成动力学接入 RL |
|---|---|
| 概率路径怎样由随机插值构造 | 奖励怎样改变终点分布的偏好 |
| 条件平均速度为何满足连续性方程 | 为什么确定性离散转移没有 PPO/GRPO 所需的普通密度 |
| ODE 与 SDE 为什么可以共享边际 | 怎样利用“同边际、不同路径”获得探索和路径似然 |
| Flow Matching、Diffusion、DDPM、DDIM 的统一关系 | Flow-GRPO、ReinFlow、FPO、RAM、$\pi_{\mathrm{RL}}$ 等路线如何取舍 |

因此，两篇文章的接缝不是重新推导概率流，而是下面这个问题：

$$
\boxed{
\underbrace{\text{ODE/SDE 共享单时刻边际}}_{\text{第一篇的结论}}
\quad\Longrightarrow\quad
\underbrace{\text{能否重写路径，使策略梯度可计算？}}_{\text{第二篇的问题}}
}
$$

对 Flow-GRPO，答案是把同边际 SDE 离散成 Gaussian 内部策略；对其他方法，答案也可能是奖励加权回归、代理似然、adjoint 或蒸馏，而不必执行 ODE-to-SDE。后文不再回顾基础生成理论，只比较这些 RL 接口在图像生成与机器人控制中的差异。

---

## 13. 图像生成与机器人控制中的差异

### 13.1 图像或视频生成

典型结构：

$$
c
\longrightarrow
(Z_0,\ldots,Z_K)
\longrightarrow
x
\longrightarrow
R(x,c).
$$

特点：

- 每条内部轨迹通常只有一个终点奖励；
- 可以并行生成同一提示的一组样本；
- 奖励可能是 OCR、GenEval、PickScore、HPS 或 VLM judge；
- 数据收集成本主要是模型推理；
- reward hacking 与审美退化是主要风险。

### 13.2 机器人 Flow Matching policy

典型结构：

$$
o_h
\longrightarrow
\text{内部 flow 生成动作块 }a_h
\longrightarrow
\text{环境转移}
\longrightarrow
r_h.
$$

特点：

- 有外层长期 credit assignment；
- 动作块本身可能由多个内部去噪步生成；
- 真实环境数据昂贵；
- 探索过大可能造成不安全动作；
- policy likelihood 是最终动作的边际密度，但很多方法优化内部路径概率的下界或代理。

### 13.3 两层随机性的区别

机器人场景中可能同时存在：

1. flow 内部采样随机性；
2. 环境转移随机性；
3. 外层策略在多个环境时刻的随机性。

所以“给 flow 加噪声”只解决内部生成策略的探索与似然问题，并不自动解决外层 RL 的样本效率、长期价值估计和安全探索。

### 13.4 $\pi_{\mathrm{RL}}$：把两层 MDP 写成一个统一策略

$\pi_{\mathrm{RL}}$ 对 $\pi_0$、$\pi_{0.5}$ 等 flow-based VLA 给出了两种在线 PPO 接口。

#### Flow-Noise：一层环境 MDP加内部路径似然

设环境时刻为 $h$，内部去噪步为 $k$。Flow-Noise 使用可学习标准差

$$
\sigma_{\theta'}(A_{h,k},o_h,t_k),
$$

构造 Gaussian 转移

$$
A_{h,k+1}
\sim
\mathcal N
\left(
A_{h,k}
\mathbin{+}
\Delta t_k
u_\theta(t_k,A_{h,k},o_h),
\sigma_{\theta'}^2(A_{h,k},o_h,t_k)\mathrm{Id}
\right).
\tag{13.1}
$$

完整去噪路径的联合对数概率可以分解为

$$
\boxed{
\log\pi_{\theta,\theta'}
(A_{h,0:K}\mid o_h)
\mathrel{=}
\log p(A_{h,0})
\mathbin{+}
\sum_{k=0}^{K-1}
\log
\pi_{\theta,\theta'}
(A_{h,k+1}\mid A_{h,k},o_h).
}
\tag{13.2}
$$

外层仍是普通机器人环境 MDP；优化时用内部路径联合概率的梯度替代难算的最终动作边际 log-likelihood。

#### Flow-SDE：显式两层 MDP

Flow-SDE 使用同边际 ODE-to-SDE 转换。内部状态和动作可写成

$$
\bar s_{h,k}
\mathrel{=}
(o_h,A_{h,k},t_k),
\qquad
\bar a_{h,k}
\mathrel{=}
A_{h,k+1}.
\tag{13.3}
$$

内部奖励为

$$
\bar r_{h,k}
\mathrel{=}
\begin{cases}
0,
&
k<K-1,
\\
R_{\mathrm{ENV}}(o_h,A_{h,K}),
&
k=K-1.
\end{cases}
\tag{13.4}
$$

内部链生成最终动作块 $A_{h,K}$，然后外层环境转移：

$$
o_{h+1}
\sim
P_{\mathrm{ENV}}
(\cdot\mid o_h,A_{h,K}).
\tag{13.5}
$$

因此，一条机器人 episode 的有效策略梯度路径同时跨越：

$$
\boxed{
\text{多个环境时刻}
\times
\text{每个环境时刻的多个去噪步。}
}
$$

#### Hybrid ODE--SDE：缩短有效 horizon

完整两层 MDP 会把 horizon 放大约 $K$ 倍。$\pi_{\mathrm{RL}}$ 因而在每个环境时刻随机选择一个内部时间执行 SDE 探索，其余内部步使用确定性 ODE。

直观上：

$$
\boxed{
\text{多数步骤负责快速确定性生成，}
\qquad
\text{少数步骤负责随机探索与策略更新。}
}
$$

论文报告这种 hybrid 两层方案相对完整两层方案约有 $2\times$ 的更新加速。它也再次说明，训练时的随机策略与部署时的确定性 ODE 可以分工。

### 13.5 $\pi_{0.7}$：RL 结果也可以通过数据和上下文进入 flow policy

$\pi_{0.7}$ 的 860M 参数 action expert 仍使用 Flow Matching 生成 50 步动作块，但论文的核心不是在线 ODE-to-SDE。

它把 episode 的速度、质量、错误标记、subgoal 图像和控制模式写入上下文，并使用这些条件训练混合质量数据。特别地，训练集包含此前 RL specialist 的 rollout，因此通用模型可以蒸馏 specialist 的高性能行为。

这给出机器人场景中的另一条闭环：

$$
\boxed{
\begin{array}{c}
\text{specialist RL 改善特定任务策略}
\\
\Downarrow
\\
\text{收集成功、失败和不同质量的自主数据}
\\
\Downarrow
\\
\text{用质量与策略上下文训练通用 flow policy}
\\
\Downarrow
\\
\text{推理时通过 prompt 选择高质量行为模式。}
\end{array}}
$$

这条路线不会替代直接在线 RL：它依赖已经收集到的经验，不能仅凭一个新的黑盒奖励立即更新策略。但它能把多个 RL specialist 的成果合并回一个通用模型，并利用失败数据提高状态覆盖与鲁棒性。

---

## 14. 一个简化的 Flow-GRPO 训练流程

给定预训练速度场 $u_{\theta_{\mathrm{ref}}}$、提示数据集 $\mathcal C$、奖励函数 $R$。

### 第一步：固定旧策略

$$
\theta_{\mathrm{old}}
\leftarrow
\theta.
$$

### 第二步：选择随机 sampler

构造

$$
b_{\theta_{\mathrm{old}}}
\mathrel{=}
u_{\theta_{\mathrm{old}}}
\mathbin{+}
\kappa s_{\theta_{\mathrm{old}}}.
$$

选择扩散率日程 $\kappa(t)$ 和离散网格。

### 第三步：为每个条件生成一组轨迹

对每个 $c\in\mathcal C$：

$$
\tau_i
\sim
\pi_{\theta_{\mathrm{old}}}(\cdot\mid c),
\qquad
i=1,\ldots,G.
$$

保存每一步：

$$
(Z_k^i,Z_{k+1}^i,t_k,c).
$$

### 第四步：计算终点奖励与组相对优势

$$
R_i=R(x^i,c),
$$

$$
\widehat A_i
\mathrel{=}
\frac{R_i-\overline R}{s_R+\varepsilon}.
$$

### 第五步：重新计算当前策略 log probability

使用同一条已采样轨迹计算

$$
\log\pi_\theta(Z_{k+1}^i\mid Z_k^i,c)
$$

和

$$
\log\pi_{\theta_{\mathrm{old}}}(Z_{k+1}^i\mid Z_k^i,c).
$$

### 第六步：优化 clipped objective

最大化式 (7.4)，通常只更新 LoRA 或部分模型参数。

### 第七步：评估真正关心的指标

除了训练奖励，还应检查：

- 未参与训练的奖励；
- 图像质量；
- 多样性；
- prompt 覆盖；
- reward hacking；
- ODE 推理时的效果；
- 不同 sampler 和不同 NFE 下的鲁棒性。

---

## 15. 实现与理论中的常见误区

### 15.1 误区：ODE 完全没有随机性

错误。

ODE 条件于初始噪声是确定的，但初始噪声随机，所以终点仍是分布。

更准确的说法是：

$$
\boxed{
\text{ODE 缺少给定中间状态后的随机分支。}
}
$$

### 15.2 误区：给 ODE 随便加 Gaussian 噪声就行

错误。

直接加噪声会在 Fokker--Planck 方程中产生扩散项，从而改变边际路径。要保持边际，需要相应的 score 漂移或其他严格构造。

### 15.3 误区：连续时间同边际保证少步 sampler 同边际

错误。

有限步离散、近似 score 和模型误差都会破坏精确等价。少步 RL 中必须单独检验 sampler bias。

### 15.4 误区：Flow-GRPO 的 SDE 是 RL 的唯一路线

错误。

它是逐步似然比策略梯度的一条自然路线，不是所有奖励优化方法的必要条件。

### 15.5 误区：内部路径似然等于终点样本似然

一般不等。

路径联合密度为

$$
p_\theta(z_{0:K}\mid c)
\mathrel{=}
p(z_0)
\prod_{k=0}^{K-1}
\pi_\theta(z_{k+1}\mid z_k,c).
\tag{15.1}
$$

终点边际密度则要积分掉全部中间变量：

$$
p_\theta(z_K\mid c)
\mathrel{=}
\int
p_\theta(z_{0:K}\mid c)
\mathrm dz_{0:K-1}.
\tag{15.2}
$$

去噪 MDP 方法使用路径上的可计算概率来构造策略梯度或变分目标，并不意味着已经免费得到精确终点似然。

### 15.6 误区：奖励提高就代表模型整体变好

错误。

奖励模型只是目标的代理。必须警惕：

- reward hacking；
- 多样性坍缩；
- 图像质量退化；
- 只在训练提示模板上过拟合；
- 机器人策略出现不安全捷径；
- 参考模型 KL 过小或过大。

---

## 16. 论文脉络

以下按“它解决了哪一个环节”组织，而不是只按时间罗列。

### 16.1 概率路径与 ODE/SDE 基础

Flow Matching for Generative Modeling 建立了用条件概率路径和条件向量场训练连续归一化流的框架。

Rectified Flow 使用线性插值，把训练目标化成速度回归。

Score-Based Generative Modeling through Stochastic Differential Equations 系统说明了 forward SDE、reverse-time SDE 与 probability-flow ODE 的关系。

这些工作共同构成本文第 2、5、8 章的基础。

### 16.2 早期奖励加权与偏好路线

Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization 绕过连续 flow 的昂贵似然，通过在线奖励加权 CFM 与 Wasserstein 正则做后训练。

Energy-Weighted Flow Matching for Offline Reinforcement Learning 和 FlowQ 一类方法把能量或 Q 值转化为 flow 训练或 guidance 信号。

这条路线说明：奖励可以通过重新加权目标分布进入 Flow Matching，而不一定先构造 SDE 策略。

### 16.3 Flow-GRPO：ODE 到 SDE 的代表路线

Flow-GRPO: Training Flow Matching Models via Online RL 提出：

1. 把确定性 ODE 改写成保持连续时间边际的 SDE；
2. 离散后获得 Gaussian 转移；
3. 把去噪过程视作 MDP；
4. 用组相对优势和 clipped policy ratio 优化；
5. 训练时减少去噪步数，推理时恢复更多步。

这篇论文最直接对应本文主线。

### 16.4 ReinFlow：机器人控制中的 Markov 噪声注入

ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning 为 flow policy 注入可学习噪声，把确定性 flow 变成离散 Gaussian Markov 过程，并推导内部路径上的策略梯度。

它强调：

- 精确的离散转移概率；
- 探索；
- 少步甚至一步 action generation；
- 连续控制与机器人任务。

它与 Flow-GRPO 共享“随机化内部 flow 以获得策略概率”的思想，但随机化构造与是否严格保持原连续边际并不相同。

### 16.5 $\pi_{\mathrm{RL}}$：VLA 中的 Flow-Noise 与 Flow-SDE

$\pi_{\mathrm{RL}}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models 把前述两种随机化方式扩展到大规模 flow-based VLA。

它给出两条并行路线：

- Flow-Noise：可学习噪声、离散路径联合 log-likelihood、一层环境 MDP；
- Flow-SDE：同边际 ODE-to-SDE、内部去噪 MDP与外层环境 MDP组成两层 MDP。

为了避免两层 MDP 把有效 horizon 放大过多，论文还提出 hybrid ODE--SDE rollout：每个环境时刻只随机选择一个去噪时间做 SDE 探索，其余步骤使用 ODE。

这篇论文把本文一直区分的三个时间尺度具体化：

$$
\boxed{
\text{环境时间 }h
\quad+\quad
\text{去噪时间 }t
\quad+\quad
\text{离散去噪步 }k.
}
$$

### 16.6 FPO：不做 ODE-to-SDE 的 on-policy 反例

Flow Matching Policy Gradients 提出 Flow Policy Optimization，用 CFM/ELBO 损失差构造 PPO 风格代理比。

它明确允许 rollout 使用确定性或随机 sampler，并避免把每个去噪步都变成 RL 动作。

它的重要意义是精确划定了本文结论的范围：

$$
\text{on-policy RL}
\quad
\text{也不必然要求}
\quad
\text{ODE-to-SDE}.
$$

### 16.7 RWFM 与 reward surrogate

Reinforcement Learning for Flow-Matching Policies 提出 Reward-Weighted Flow Matching，以及用 learned reward surrogate 和组相对优势加权 Flow Matching 损失的方法。

它使用显式动作扰动器扩展探索，而不是必须在内部 ODE 每一步建立 SDE likelihood。

### 16.8 CPS、GLASS 与 PRECISE：重新审视随机 sampler

Coefficients-Preserving Sampling 指出少步 Flow-SDE 可能产生过量噪声，并提出保持插值系数的随机采样规则。

GLASS Flows 进一步说明，随机 Markov 转移不一定只能由外层 SDE 模拟；可以构造“flow 中的 flow”，用内层随机初值加确定性 ODE 来采样条件转移。

PRECISE 指出：

- Euler SDE 在少步下可能过量加噪；
- CPS 可能收缩后验不确定性并造成边际偏差；
- sampler 应同时满足合理探索日程与有限步 SDE 一致性。

这几篇工作把问题从“要不要随机化”推进到“怎样随机化才不破坏预训练动力学”。

### 16.9 RAM：回到预训练式回归

Reinforce Adjoint Matching 从 KL 正则化奖励倾斜分布出发，把 RL 最优控制条件转化为奖励修正的回归目标。

它用 ODE 生成终点，再解析加噪构造训练状态，不需要 SDE rollout。

这条路线重新连接了 Flow Matching 最初的优势：

$$
\boxed{
\text{把复杂生成学习还原成随机样本上的平方回归。}
}
$$

### 16.10 $\pi_{0.7}$：从 RL specialist 到可提示的通用 flow policy

$\pi_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities 不是新的在线 RL 算法，但它展示了 RL 成果进入 Flow Matching 通用策略的另一种机制。

其 action expert 使用 Flow Matching，并在训练上下文中加入：

- episode quality；
- episode speed；
- mistake label；
- subtask language；
- subgoal images；
- control mode。

训练数据包含旧策略的自主 rollout、失败轨迹以及 RL specialist 的经验。借助质量与策略上下文，模型不会被迫无条件平均成功、失败和不同速度的行为；推理时则可以提示高质量、无错误的行为模式。

所以它与本文的关系不是

$$
\text{ODE}\rightarrow\text{SDE}\rightarrow\text{online RL},
$$

而是

$$
\boxed{
\text{RL 产生经验}
\rightarrow
\text{质量条件标注}
\rightarrow
\text{Flow Matching 蒸馏}
\rightarrow
\text{通用模型继承 specialist 能力。}
}
$$

---

## 17. 结论：ODE-to-SDE 是一种 RL 接口

第一篇解释了 Flow Matching 如何定义生成分布；第二篇关心的是怎样用奖励改变这个分布。对 KL 正则化奖励目标，理想终点是

$$
\rho_{\mathrm{ref}}
\longrightarrow
\rho^*
\propto
\rho_{\mathrm{ref}}
\exp(R/\beta).
$$

如果选择 Flow-GRPO 这类逐步策略梯度，关键链条是

$$
\boxed{
\text{ODE 概率流}
\Longrightarrow
\text{score 修正漂移}
\Longrightarrow
\text{SDE 保持同一边际}
\Longrightarrow
\text{离散 Gaussian 转移}
\Longrightarrow
\text{PPO/GRPO 可计算。}
}
$$

但选择 RL 算法时，首先应判断是否真的需要精确的逐步策略似然比：

$$
\boxed{
\begin{array}{c}
\text{是否必须使用黑盒奖励的严格逐步策略梯度？}
\\
\begin{cases}
\text{是}
&
\rightarrow
\text{随机 Markov 转移，常用同边际 SDE；}
\\
\text{否}
&
\rightarrow
\text{继续判断奖励是否可微、是否有离线数据、}
\\
&
\phantom{\rightarrow}
\text{是否接受代理似然或奖励加权回归。}
\end{cases}
\end{array}}
$$

所以最准确的一句话是：**从 ODE 到 SDE，不是 Flow Matching 接受奖励信号的数学前提，而是把确定性生成路径接成可探索、可计算逐步似然的 PPO/GRPO 策略接口。**

---

## 参考文献

1. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR 2023.
2. Xingchao Liu, Chengyue Gong, Qiang Liu. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003). 2022.
3. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). ICLR 2021.
4. Kevin Black et al. [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). 2023.
5. Jiajun Fan et al. [Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization](https://arxiv.org/abs/2502.06061). ICLR 2025.
6. Shiyuan Zhang, Weitong Zhang, Quanquan Gu. [Energy-Weighted Flow Matching for Offline Reinforcement Learning](https://arxiv.org/abs/2503.04975). 2025.
7. Jie Liu et al. [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470). 2025.
8. Tonghe Zhang, Chao Yu, Sichang Su, Yu Wang. [ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](https://arxiv.org/abs/2505.22094). 2025.
9. Samuel Pfrommer, Yixiao Huang, Somayeh Sojoudi. [Reinforcement Learning for Flow-Matching Policies](https://arxiv.org/abs/2507.15073). 2025.
10. David McAllister et al. [Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053). 2025.
11. Feng Wang, Zihao Yu. [Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching](https://arxiv.org/abs/2509.05952). 2025.
12. Peter Holderrieth et al. [GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models](https://arxiv.org/abs/2509.25170). ICLR 2026.
13. Jade Zou et al. [PRECISE: SDE-Consistent Stochastic Sampling for RL Post-Training of Flow-Matching Models](https://arxiv.org/abs/2605.23522). 2026.
14. Andreas Bergmeister et al. [Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models](https://arxiv.org/abs/2605.10759). 2026.
15. Boshu Lei, Kostas Daniilidis, Antonio Loquercio. [Reinforcement Learning for Flow-Matching Policies with Density Transport](https://arxiv.org/abs/2606.08602). 2026.
16. Kang Chen et al. [$\pi_{\mathrm{RL}}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models](https://arxiv.org/abs/2510.25889). 2025.
17. Physical Intelligence et al. [$\pi_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://arxiv.org/abs/2604.15483). 2026.

> **文献范围说明。**
>
> 本文调研截至 2026 年 8 月 2 日。2025—2026 年的 Flow Matching RL 仍在快速演化，部分工作是 arXiv 预印本或 work in progress。正文把已经发表的连续时间理论、论文自述结论和后续论文对早期 sampler 的修正分开表述，避免把早期方法的连续时间正确性误写成有限步实现的无条件正确性。

