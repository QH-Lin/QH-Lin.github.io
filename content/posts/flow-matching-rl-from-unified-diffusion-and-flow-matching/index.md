---
title: "流匹配的RL（一）：统一理解Diffusion与Flow-Matching及ODE、SDE"
date: 2026-07-31T17:39:06+08:00
draft: false
tags:
  - flow matching
  - diffusion models
  - reinforcement learning
categories:
  - AI
description: "从随机插值与概率路径出发，统一理解 Diffusion、Flow Matching、ODE 与 SDE。"
showToc: true
TocOpen: false
math: true
---

## 从双端点随机插值出发，将ODE与SDE统一

本文核心：

给定一个容易采样的噪声分布和一个目标数据分布，怎样先设计连接二者的概率路径，直接得到概率流 ODE，再额外构造同边际 SDE，并把它们与 Flow Matching、DDPM 和 DDIM 联系起来？

本文以《Stochastic Interpolants: A Unifying Framework for Flows and Diffusions》为主，结合附录里的多篇文章以及gpt，整理而成。

> ODE 和 SDE 都是描述“一个状态如何随时间变化”的方程，区别在于有没有随机噪声。
ODE：Ordinary Differential Equation，常微分方程。
SDE：Stochastic Differential Equation，随机微分方程。

$$
\boxed{
\text{随机插值、ODE 和 SDE 的单条轨迹通常不同，}
\quad
\text{但它们可以具有相同的单时刻边际分布。}
}
$$

![图 1：Stochastic Interpolant 的总体框架](images/paper-figure-01-framework.png)

*说明：先用随机插值规定从 $\rho_0$ 到 $\rho_1$ 的边际路径，再学习速度与 score；ODE 和带 score 修正的 SDE 是实现同一组边际的不同生成动力学。图中的 SDE 不是由单个 latent noise $z$ 直接变成的，而是引入布朗运动后另外构造的。来源：[Albergo, Boffi & Vanden-Eijnden, v4, Figure 1](https://arxiv.org/abs/2303.08797v4)。*

> **备注：单时刻边际的含义**
>
> 一个随机过程包含整条轨迹 $\{X_t:0\le t\le1\}$。完整的过程规律还包括多个时刻之间的联合分布，例如 $(X_{0.2},X_{0.8})$。
>
> 单时刻边际只固定一个 $t$，看 $X_t$ 的分布：
>
> $$
> X_t\sim q(t,\cdot).
> $$
>
> “边际”来自把联合分布中的其他变量积分掉。例如若 $(X_s,X_t)$ 的联合密度是 $p_{s,t}(x,y)$，那么
>
> $$
> p_t(y)
> \mathrel{=}
> \int_{\mathbb R^d}
> p_{s,t}(x,y)\,\mathrm dx
> $$
>
> 就是 $X_t$ 的边际密度。两个过程可以在每个时刻具有相同边际，却有完全不同的 $p_{s,t}$，因而轨迹形状和多时刻联合分布都可以不同。

---

# 第一部分：先设计一条概率路径

## 1. 三类对象必须分开

本文会同时出现三类随机对象。

| 对象 | 记号 | 密度 | 作用 |
|---|---|---|---|
| 训练时设计的随机插值 | $x_t$ | $\rho(t,x)$ | 规定希望模型复现的概率路径 |
| 生成 ODE | $X_t$ | $q(t,x)$ | 确定性地生成样本 |
| 生成 SDE | $Z_t$ | $p(t,x)$ | 随机地生成样本 |

还要区分两类速度：

| 记号 | 含义 |
|---|---|
| $Y_t=\dot x_t$ | 某一条随机插值轨迹的随机速度 |
| $b(t,x)=\mathbb E[Y_t\mid x_t=x]$ | 给定时间和位置后的条件平均速度 |

其中 $Y_t$ 依赖隐藏的端点和高斯变量；$b(t,x)$ 只依赖当前的 $(t,x)$，因此可以放进 ODE 或 SDE。

---

## 2. 双端点随机插值

### 2.1 端点分布与耦合

本文先使用生成时间

$$
t:0\longrightarrow1,
$$

并约定

$$
\rho_0=\text{容易采样的基分布，通常是高斯噪声},
$$

$$
\rho_1=\text{目标数据分布}.
$$

采样一对端点

$$
(x_0,x_1)\sim\nu,
$$

其中 $\nu$ 的两个边际分别是 $\rho_0$ 和 $\rho_1$。

最简单的选择是独立耦合

$$
\nu=\rho_0\otimes\rho_1.
$$

这表示随机选择一个噪声样本，再独立选择一个数据样本，把它们临时配成一对。

耦合 $\nu$ 不改变端点分布，但会改变中间路径和学习难度。

### 2.2 一般双端点公式

选择一个满足

$$
I(0,x_0,x_1)=x_0,
\qquad
I(1,x_0,x_1)=x_1
$$

的确定性插值函数。

再独立采样

$$
z\sim\mathcal N(0,\mathrm{Id}),
\qquad
z\perp(x_0,x_1),
$$

定义

$$
\boxed{
x_t
\mathrel{=}
I(t,x_0,x_1)+\gamma(t)z.
}
\tag{2.1}
$$

对一般 stochastic interpolant，取

$$
\gamma(0)=\gamma(1)=0,
$$

并通常要求

$$
\gamma(t)>0,
\qquad 0<t<1.
$$

由式 (2.1) 和 $\gamma(0)=\gamma(1)=0$ 可知

$$
x_{t=0}=x_0\sim\rho_0,
\qquad
x_{t=1}=x_1\sim\rho_1.
$$

定义 $x_t$ 的单时刻边际密度

$$
\boxed{
x_t\sim\rho(t,\cdot).
}
\tag{2.2}
$$

式 (2.1) 的首要作用不是直接生成数据，而是规定一条连接 $\rho_0$ 和 $\rho_1$ 的概率路径。

> **备注：式 (2.1) 通常不能直接用于生成的原因**
>
> 训练时可以同时取得噪声端点 $x_0$ 和数据端点 $x_1$，所以能够直接计算 $x_t$。
>
> 生成时只有新的 $x_0\sim\rho_0$，并不知道它应该对应哪个真实数据 $x_1$。因此需要学习一个只依赖 $(t,x)$ 的速度场，再从 $x_0$ 出发运行 ODE 或 SDE。

### 2.3 本文使用的正则性约定

为了集中讨论主线，后文假设：

- $I$ 和 $\gamma$ 对时间足够光滑；
- 所需期望有限；
- 可以交换时间求导、期望和积分；
- 相关密度和速度场足够光滑；
- 在使用 score 时，内部时刻的密度为正。

这些假设保证后面的链式法则、分部积分、Fokker--Planck 方程和 PDE 唯一性可以正常使用。

当 $\gamma(t)>0$ 时，中间分布包含高斯卷积（把原来的概率分布用高斯噪声轻轻模糊、摊开），通常会比 $\gamma=0$ 的路径更平滑。

![图 2：latent noise 尺度 $\gamma(t)$ 对中间边际的影响](images/paper-figure-04-gamma-density.png)

*说明：每一行使用不同的 $\gamma(t)$。额外高斯变量越能平滑中间密度，端点模态在中间时刻产生的重叠和伪结构通常越少；但这只是路径设计的效果，并不等于生成 SDE 的布朗噪声。来源：[论文 v4, Figure 4](https://arxiv.org/abs/2303.08797v4)。*


---

## 3. 两种不同的“噪声旋钮”

后文会出现两种噪声，它们不能混为一谈。

| 噪声 | 出现在哪里 | 主要作用 | 是否直接使生成轨迹随机 |
|---|---|---|---|
| $\gamma(t)z$ | 训练时的随机插值 | 设计和平滑中间边际，改变监督信号 | 否 |
| $\sqrt{2\epsilon(t)}\,\mathrm dW_t$ | 生成 SDE | 改变生成动力学和轨迹随机性 | 是 |

$\gamma$ 决定“训练时希望学习哪条边际路径”；$\epsilon$ 决定“生成时用多随机的 SDE 实现这条边际路径”。

这正是统一框架的重要自由度：

$$
\boxed{
\text{先选概率路径，再选实现这条路径的动力学。}
}
$$

---

# 第二部分：从随机插值得到可学习的场

## 4. 从随机轨迹速度得到确定速度场

对式 (2.1) 关于时间求导：

$$
\boxed{
Y_t:=\dot x_t
\mathrel{=}
\partial_tI(t,x_0,x_1)+\dot\gamma(t)z.
}
\tag{4.1}
$$

$Y_t$ 是随机的。即使给定 $x_t=x$，也可能有很多不同的 $(x_0,x_1,z)$ 经过同一个位置，并给出不同的速度。

ODE 在同一个 $(t,x)$ 处只能使用一个确定速度。因此定义

$$
\boxed{
b(t,x)
:=
\mathbb E[Y_t\mid x_t=x].
}
\tag{4.2}
$$

将式 (4.1) 代入式 (4.2) 可知

$$
b(t,x)
\mathrel{=}
\mathbb E[
\partial_tI(t,x_0,x_1)+\dot\gamma(t)z
\mid x_t=x
].
$$

一般并没有逐样本等式

$$
Y_t=b(t,x_t).
$$

由式 (4.2) 的条件期望定义可知，正确的关系是

$$
\boxed{
\mathbb E[Y_t\mid x_t]=b(t,x_t).
}
\tag{4.3}
$$

> **备注：条件期望的直观理解**
>
> 可以把所有训练样本按照当前的 $(t,x_t)$ 分组。在同一组中，隐藏端点和随机速度可以不同。$b(t,x)$ 就是这一组随机速度的平均值。
>
> 因而，$b$ 是从“许多微观随机速度”提取出的“宏观概率输运速度”。

---

## 5. 为什么 $\rho$ 满足连续性方程

取任意光滑且只在有限区域内非零的测试函数

$$
\varphi\in C_c^\infty(\mathbb R^d).
$$

> **备注：测试函数的定义与引入原因**
>
> $C^\infty$ 表示函数可以无限次求导；下标 $c$ 表示 compact support，即它只在某个有限区域内非零。可以把 $\varphi$ 看成一个“光滑探测器”：如果它在区域 $A$ 内接近 $1$、区域外接近 $0$，那么
>
> $$
> \mathbb E[\varphi(x_t)]
> $$
>
> 就近似表示 $x_t$ 落入区域 $A$ 的概率。选取紧支撑函数还有一个技术好处：做分部积分时，无穷远处的边界项自动消失。

> **备注：三个空间微分算子的含义**
>
> 对标量函数 $f:\mathbb R^d\to\mathbb R$，梯度是向量
>
> $$
> \boxed{
> \nabla f
> \mathrel{=}
> \left(
> \partial_{x_1}f,\ldots,\partial_{x_d}f
> \right).
> }
> $$
>
> 它指向函数上升最快的方向。对向量场 $F=(F_1,\ldots,F_d)$，散度是标量
>
> $$
> \boxed{
> \nabla\cdot F
> \mathrel{=}
> \sum_{i=1}^d\partial_{x_i}F_i.
> }
> $$
>
> 散度衡量当前位置附近的“净流出”：$\nabla\cdot F>0$ 表示流出多于流入。Laplace 算子是
>
> $$
> \boxed{
> \Delta f
> \mathrel{=}
> \sum_{i=1}^d\partial_{x_i}^2f
> \mathrel{=}
> \nabla\cdot(\nabla f).
> }
> $$
>
> 因此可以记成
>
> $$
> \nabla:\text{标量}\to\text{向量},
> \qquad
> \nabla\cdot:\text{向量}\to\text{标量},
> \qquad
> \Delta=\nabla\cdot\nabla.
> $$
>
> 直观地说，梯度描述“往哪里升高”，散度描述“是否净流出”，Laplace 算子描述一个标量场在当前位置相对于周围是凸起还是凹陷。

> **备注：梯度相同但 Laplace 算子符号仍可相反的原因**
>
> 考虑下面两个一维温度场：
>
> $$
> f_1(x)=x^2,
> \qquad
> f_2(x)=-x^2+4x-2.
> $$
>
> 在 $x=1$ 处，两者的函数值和梯度完全相同：
>
> $$
> f_1(1)=f_2(1)=1,
> \qquad
> f_1'(1)=f_2'(1)=2.
> $$
>
> 但它们的 Laplace 算子（在一维中就是二阶导数）符号相反：
>
> $$
> \Delta f_1=f_1''=2,
> \qquad
> \Delta f_2=f_2''=-2.
> $$
>
> 梯度相同意味着：在 $x=1$ 处，两个温度场向右都以相同的局部斜率升高。由 Fourier 热传导定律
>
> $$
> J=-k\nabla f,
> $$
>
> 两者在该点的热流密度都向左，大小也相同。然而，一个点上的梯度只描述该点的坡度，不能告诉我们附近的热流是否平衡；为此还要考察梯度在空间中怎样变化。
>
> **情况一：$\Delta f_1>0$。** 在 $x=1$ 附近令 $x=1+h$，则
>
> $$
> f_1(1+h)=1+2h+h^2.
> $$
>
> 梯度场为 $\nabla f_1=2x$，从左向右越来越大：
>
> ```text
> x=1-ε           x=1           x=1+ε
>  2-2ε    →        2     →       2+2ε   →
> ```
>
> 因而梯度场的右端流出多于左端流入：
>
> $$
> \nabla\cdot(\nabla f_1)=\Delta f_1=2>0.
> $$
>
> 真实热流 $J_1=-2kx$ 与梯度方向相反，所以从右边流入的热量多于从左边流出的热量，$x=1$ 附近会积累热量。在热容等常数已归入系数 $k$ 的写法下，
>
> $$
> \frac{\partial f_1}{\partial t}
> =k\Delta f_1>0,
> $$
>
> 即温度上升。
>
> **情况二：$\Delta f_2<0$。** 同样在 $x=1$ 附近，
>
> $$
> f_2(1+h)=1+2h-h^2.
> $$
>
> 梯度场为 $\nabla f_2=-2x+4$，从左向右越来越小：
>
> ```text
> x=1-ε           x=1           x=1+ε
>  2+2ε    →        2     →       2-2ε   →
> ```
>
> 虽然 $x=1$ 处的梯度仍向右且大小仍为 $2$，但左边的梯度比右边大，因此梯度场有净流入：
>
> $$
> \nabla\cdot(\nabla f_2)=\Delta f_2=-2<0.
> $$
>
> 对应的真实热流向左，且从左端流出的热量多于从右端流入的热量，所以
>
> $$
> \frac{\partial f_2}{\partial t}
> =k\Delta f_2<0,
> $$
>
> 即温度下降。
>
> | 量 | 两个场在 $x=1$ 处的情况 | 物理含义 |
> |---|---|---|
> | $f$ | 相同 | 当前温度相同 |
> | $\nabla f$ | 都等于 $2$ | 温度都向右升高，热流都向左 |
> | $\Delta f$ | 一个为 $+2$，一个为 $-2$ | 一个会升温，一个会降温 |
>
> 所以，梯度描述一点处的“坡度和方向”；Laplace 算子描述附近的坡度如何变化，从而决定该点附近是净流入还是净流出。

沿一条随机插值轨迹使用链式法则：

$$
\frac{\mathrm d}{\mathrm dt}\varphi(x_t)
\mathrel{=}
\nabla\varphi(x_t)\cdot Y_t.
$$

> **备注：多变量链式法则产生该点积的原因**
>
> 写成坐标形式，$x_t=(x_t^{(1)},\ldots,x_t^{(d)})$。普通多变量链式法则给出
>
> $$
> \frac{\mathrm d}{\mathrm dt}\varphi(x_t)
> \mathrel{=}
> \sum_{i=1}^d
> \frac{\partial\varphi}{\partial x_i}(x_t)
> \frac{\mathrm dx_t^{(i)}}{\mathrm dt}.
> $$
>
> 第一组分量组成 $\nabla\varphi(x_t)$，第二组分量组成 $\dot x_t=Y_t$，所以求和就是点积。虽然 $x_t$ 是随机的，但固定一次 $(x_0,x_1,z)$ 后，它就是一条普通的可微曲线，因此可以逐条轨迹使用链式法则。

取期望：

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(x_t)]
\mathrel{=}
\mathbb E[
\nabla\varphi(x_t)\cdot Y_t
].
\tag{5.1}
$$

> **备注：时间导数移入期望的依据**
>
> 这里使用
>
> $$
> \frac{\mathrm d}{\mathrm dt}\mathbb E[\varphi(x_t)]
> \mathrel{=}
> \mathbb E\!\left[
> \frac{\mathrm d}{\mathrm dt}\varphi(x_t)
> \right].
> $$
>
> 直观上是“许多轨迹观测值的平均变化速度”等于“每条轨迹观测值变化速度的平均”。严格成立需要可微性、可积性以及一个可积函数控制导数；本文第 2.3 节的正则性约定正是为了允许这一步。没有这些条件时，求导与期望不能随意交换。

由式 (5.1) 和式 (4.2) 可知，对 $x_t$ 做条件平均后有

$$
\begin{aligned}
\mathbb E[
\nabla\varphi(x_t)\cdot Y_t
]
&=
\mathbb E[
\nabla\varphi(x_t)\cdot
\mathbb E[Y_t\mid x_t]
]
\\
&=
\mathbb E[
\nabla\varphi(x_t)\cdot b(t,x_t)
]
\\
&=
\int_{\mathbb R^d}
\nabla\varphi(x)\cdot b(t,x)\rho(t,x)\,\mathrm dx.
\end{aligned}
\tag{5.2}
$$

> **备注：式 (5.2) 前两个等号的依据**
>
> 条件期望 $\mathbb E[Y_t\mid x_t]$ 可以理解为：按照 $x_t$ 的取值把样本分组，再在每组内部平均 $Y_t$。首先使用塔式法则
>
> $$
> \mathbb E[Z]
> \mathrel{=}
> \mathbb E\!\left[\mathbb E[Z\mid x_t]\right].
> $$
>
> 然后注意：给定 $x_t$ 后，$\nabla\varphi(x_t)$ 已经是已知量，可以从条件期望中提出：
>
> $$
> \mathbb E[
> \nabla\varphi(x_t)\cdot Y_t
> \mid x_t]
> \mathrel{=}
> \nabla\varphi(x_t)\cdot
> \mathbb E[Y_t\mid x_t].
> $$
>
> 最后代入 $b(t,x)=\mathbb E[Y_t\mid x_t=x]$。这一步解释了为什么随机的微观速度 $Y_t$ 最终变成只依赖 $(t,x)$ 的确定速度场 $b$。

对式 (5.2) 的空间变量分部积分，得到

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(x_t)]
\mathrel{=}
-\int_{\mathbb R^d}
\varphi(x)\nabla\cdot(\rho b)(t,x)\,\mathrm dx.
\tag{5.3}
$$

> **备注：分部积分产生负号的原因**
>
> 分部积分把作用在测试函数 $\varphi$ 上的空间导数转移到概率流量 $\rho b$ 上。先看一维。由乘积求导公式
>
> $$
> (\varphi F)'=\varphi'F+\varphi F',
> $$
>
> 在整个实数轴上积分可得
>
> $$
> \int_{\mathbb R}\varphi'(x)F(x)\,\mathrm dx
> \mathrel{=}
> [\varphi(x)F(x)]_{-\infty}^{+\infty}
> \mathbin{-}
> \int_{\mathbb R}\varphi(x)F'(x)\,\mathrm dx.
> $$
>
> 这里的负号来自把含 $F'$ 的积分移到等号右边。由于测试函数 $\varphi\in C_c^\infty$ 具有紧支撑，它在足够远处为零，边界项也随之消失，因此
>
> $$
> \int_{\mathbb R}\varphi'(x)F(x)\,\mathrm dx
> \mathrel{=}
> -\int_{\mathbb R}\varphi(x)F'(x)\,\mathrm dx.
> $$
>
> 在 $d$ 维中，对每个坐标分别使用这个公式并求和，普通导数就变成散度。令向量场 $F=\rho b$，得到
>
> $$
> \boxed{
> \int_{\mathbb R^d}
> \nabla\varphi\cdot(\rho b)\,\mathrm dx
> \mathrel{=}
> -\int_{\mathbb R^d}
> \varphi\,\nabla\cdot(\rho b)\,\mathrm dx.
> }
> $$
>
> 直观上，$\rho b$ 是概率流量；$\nabla\cdot(\rho b)>0$ 表示某处流出多于流入，所以该处密度应当下降。这与式 (5.3) 中的负号一致。

另一方面，

$$
\mathbb E[\varphi(x_t)]
\mathrel{=}
\int_{\mathbb R^d}\varphi(x)\rho(t,x)\,\mathrm dx,
$$

所以

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(x_t)]
\mathrel{=}
\int_{\mathbb R^d}
\varphi(x)\partial_t\rho(t,x)\,\mathrm dx.
\tag{5.4}
$$

> **备注：式 (5.4) 成立的原因**
>
> 因为 $x_t$ 的密度是 $\rho(t,x)$，任意函数的期望都可以按密度积分：
>
> $$
> \mathbb E[\varphi(x_t)]
> \mathrel{=}
> \int_{\mathbb R^d}
> \varphi(x)\rho(t,x)\,\mathrm dx.
> $$
>
> 这里 $x$ 只是积分变量，而 $\varphi(x)$ 本身不依赖时间。因此
>
> $$
> \begin{aligned}
> \frac{\mathrm d}{\mathrm dt}
> \mathbb E[\varphi(x_t)]
> &=
> \frac{\mathrm d}{\mathrm dt}
> \int\varphi(x)\rho(t,x)\,\mathrm dx
> \\
> &=
> \int\varphi(x)\partial_t\rho(t,x)\,\mathrm dx.
> \end{aligned}
> $$
>
> 如果测试函数也写成 $\varphi(t,x)$，还会额外出现 $\int(\partial_t\varphi)\rho\,\mathrm dx$；本文选择与时间无关的 $\varphi$，所以没有这一项。

比较式 (5.3) 与式 (5.4) 可知

$$
\int_{\mathbb R^d}
\varphi(x)
\left[
\partial_t\rho+\nabla\cdot(\rho b)
\right](t,x)\,\mathrm dx
=0.
$$

由于测试函数可以任意选择，

$$
\boxed{
\partial_t\rho(t,x)
\mathbin{+}
\nabla\cdot\bigl(\rho(t,x)b(t,x)\bigr)
=0.
}
\tag{5.5}
$$

这就是连续性方程。

> **备注：由“对任意测试函数积分为零”推出括号内函数为零的依据**
>
> 令
>
> $$
> h(t,x)
> \mathrel{=}
> \partial_t\rho(t,x)+\nabla\cdot(\rho b)(t,x).
> $$
>
> 如果某个区域内 $h$ 明显为正，就可以选择一个在该区域内为正、其他地方为零的光滑测试函数 $\varphi$，此时 $\int\varphi h\,\mathrm dx$ 会大于零，与“对所有 $\varphi$ 都等于零”矛盾。$h$ 为负的情况同理。因此 $h=0$，至少在几乎处处或分布意义下成立。这种先与任意测试函数积分的证明叫作弱形式证明。

> **备注：连续性方程的含义**
>
> $\rho b$ 是概率流量。某个区域的概率密度下降，是因为从该区域流出的概率多于流入的概率。
>
> 因此
>
> $$
> \text{局部密度变化}
> \mathbin{+}
> \text{局部净流出}
> =0.
> $$

---

## 6. score 从哪里来

定义中间密度的 score：

$$
\boxed{
s(t,x)
:=
\nabla_x\log\rho(t,x).
}
\tag{6.1}
$$

对于

$$
x_t=I(t,x_0,x_1)+\gamma(t)z,
$$

固定 $(x_0,x_1)$ 后，$x_t$ 是一个高斯随机变量：

$$
x_t\mid(x_0,x_1)
\sim
\mathcal N\left(
I(t,x_0,x_1),
\gamma^2(t)\mathrm{Id}
\right).
$$

该条件高斯的 score 是

$$
-\frac{x-I(t,x_0,x_1)}{\gamma^2(t)}.
$$

> **备注：高斯分布的 score 为什么是这个形式？**
>
> 对一般各向同性高斯
>
> $$
> p(x)
> \mathrel{=}
> \frac{1}{(2\pi\alpha^2)^{d/2}}
> \exp\!\left(
> -\frac{\|x-\mu\|^2}{2\alpha^2}
> \right),
> $$
>
> 先取对数：
>
> $$
> \log p(x)
> \mathrel{=}
> \text{与 }x\text{ 无关的常数}
> \mathbin{-}
> \frac{\|x-\mu\|^2}{2\alpha^2}.
> $$
>
> 再对 $x$ 求梯度：
>
> $$
> \nabla_x\log p(x)
> \mathrel{=}
> -\frac{x-\mu}{\alpha^2}
> \mathrel{=}
> \frac{\mu-x}{\alpha^2}.
> $$
>
> 所以高斯 score 总是指向均值 $\mu$，并且离均值越远，向中心的指向越强。这里令 $\mu=I(t,x_0,x_1)$、$\alpha=\gamma(t)$，就得到上式。

在随机样本 $x=x_t$ 上，

$$
x_t-I(t,x_0,x_1)=\gamma(t)z,
$$

所以条件 score 标签可以写成

$$
-\frac{z}{\gamma(t)}.
$$

对隐藏端点和高斯变量做条件平均，得到边际 score：

$$
\boxed{
s(t,x)
\mathrel{=}
-\frac1{\gamma(t)}
\mathbb E[z\mid x_t=x],
\qquad 0<t<1.
}
\tag{6.2}
$$

> **备注：score 的物理意义是什么？**
>
> 固定时刻 $t$，可以把 $\rho(t,x)$ 看成空间中的一张概率密度地形。由
>
> $$
> s(t,x)
> \mathrel{=}
> \nabla_x\log\rho(t,x)
> \mathrel{=}
> \frac{\nabla_x\rho(t,x)}{\rho(t,x)},
> $$
>
> score 的方向是当前位置上对数密度增长最快的方向，也就是局部指向“更可能出现样本”的区域。因此，沿着 $s$ 移动会使密度在一阶近似下增加得最快；


定义 denoiser

$$
\boxed{
\eta_z(t,x)
:=
\mathbb E[z\mid x_t=x].
}
\tag{6.3}
$$

则

$$
\boxed{
s(t,x)=-\frac{\eta_z(t,x)}{\gamma(t)}.
}
\tag{6.4}
$$

> **备注：为什么叫 denoiser？**
>
> 网络看到被高斯变量扰动后的 $x_t$，并预测产生该样本的平均高斯变量。知道平均噪声后，就能反推出更干净的中心位置或数据端点估计。

---

## 7. 速度分解

除了总速度 $b$，再定义只对确定性插值部分求条件平均的速度

$$
\boxed{
v(t,x)
:=
\mathbb E[
\partial_tI(t,x_0,x_1)
\mid x_t=x
].
}
\tag{7.1}
$$

将式 (4.1) 代入式 (4.2)，再使用式 (7.1) 可知

$$
b(t,x)
\mathrel{=}
v(t,x)
\mathbin{+}
\dot\gamma(t)\mathbb E[z\mid x_t=x].
$$

再由式 (6.2) 可知

$$
\mathbb E[z\mid x_t=x]
\mathrel{=}
-\gamma(t)s(t,x),
$$

代入上式，得到

$$
\boxed{
b(t,x)
\mathrel{=}
v(t,x)
\mathbin{-}
\gamma(t)\dot\gamma(t)s(t,x).
}
\tag{7.2}
$$

这是连接 velocity、score 和 latent noise 的关键公式。

它说明：

- $v$ 描述确定性插值中心的平均移动；
- $-\gamma\dot\gamma s$ 描述插值噪声尺度变化对概率流的贡献；
- 真正放进概率流 ODE 的速度是二者之和 $b$。

> **备注：区分 $b$ 和 $v$ 的原因**
>
> 当 $\gamma$ 随时间变化时，只学习 $\partial_tI$ 的条件平均并不能得到完整边际速度。高斯扰动本身的收缩或扩张也会推动密度变化，因此还需要 score 修正项。
>
> 当 $\gamma=0$ 时，式 (7.2) 退化为 $b=v$，这正是后面的 endpoint Flow Matching。

---

## 8. 为什么平方回归能够学习这些场

### 8.1 学习总速度 $b$

由式 (4.1) 可知随机速度标签。训练网络 $b_\theta(t,x)$ 时使用

$$
\boxed{
\mathcal L_b(\theta)
\mathrel{=}
\mathbb E
\left\|
b_\theta(t,x_t)
\mathbin{-}
\left[
\partial_tI(t,x_0,x_1)+\dot\gamma(t)z
\right]
\right\|^2.
}
\tag{8.1}
$$

平方损失的最优预测是条件期望。因此由式 (4.2) 可知，在理想函数类中，

$$
b_\theta^*(t,x)
\mathrel{=}
\mathbb E[Y_t\mid x_t=x]
\mathrel{=}
b(t,x).
$$

> **备注：平方损失的最优解为条件平均的原因**
>
> 关键是网络只看到 $(t,x_t)$，看不到产生该样本的隐藏端点 $(x_0,x_1)$ 和高斯变量 $z$。固定时间 $t$ 和位置 $x_t=x$ 后，网络只能输出一个确定向量。把这个待选择的输出记为 $a$。
>
> 在所有满足 $x_t=x$ 的训练样本中，随机速度标签
>
> $$
> Y_t
> \mathrel{=}
> \partial_tI(t,x_0,x_1)+\dot\gamma(t)z
> $$
>
> 可能各不相同。对于当前位置，网络要选择 $a$，使这些标签到 $a$ 的条件平均平方距离
>
> $$
> \ell(a)
> :=
> \mathbb E\!\left[
> \|a-Y_t\|^2
> \mid x_t=x
> \right]
> $$
>
> 尽可能小。对向量 $a$ 求梯度：
>
> $$
> \begin{aligned}
> \nabla_a\ell(a)
> &=
> \mathbb E\!\left[
> 2(a-Y_t)
> \mid x_t=x
> \right]
> \\
> &=
> 2\left(
> a-\mathbb E[Y_t\mid x_t=x]
> \right).
> \end{aligned}
> $$
>
> 令梯度等于零，得到
>
> $$
> \boxed{
> a^*
> \mathrel{=}
> \mathbb E[Y_t\mid x_t=x]
> \mathrel{=}
> b(t,x).
> }
> $$
>
> 这是最小值，因为平方距离关于 $a$ 是开口向上的凸二次函数。因此，令 $a=b_\theta(t,x)$ 并对所有时间和位置共同训练，平方回归就会把同一 $(t,x)$ 对应的随机速度标签取条件平均，最终得到确定速度场 $b(t,x)$。
>
> 一维直观例子：假设同一个 $(t,x)$ 对应的三个速度标签分别是 $8,10,12$。网络面对相同输入只能输出同一个数。输出它们的平均值 $10$，会使平方距离之和
>
> $$
> (a-8)^2+(a-10)^2+(a-12)^2,
> $$
>
> 达到最小。条件期望 $\mathbb E[Y_t\mid x_t=x]$ 就是连续随机情形下的这种“分组后取平均”。

### 8.2 学习 denoiser 和 score

数值上更稳定的做法通常是学习

$$
\boxed{
\mathcal L_\eta(\theta)
\mathrel{=}
\mathbb E
\left\|
\eta_\theta(t,x_t)-z
\right\|^2.
}
\tag{8.2}
$$

由式 (6.3) 和平方损失的条件期望性质可知，其理想最优解是

$$
\eta_\theta^*(t,x)
\mathrel{=}
\mathbb E[z\mid x_t=x]
\mathrel{=}
\eta_z(t,x).
$$

同样的平方损失分解说明：噪声网络学到的不是某一次采样中的具体 $z$，而是在观察到 $(t,x_t=x)$ 后最合理的平均噪声。单个 $x_t$ 可能由多个数据端点和多个噪声组合产生，网络只能对这些可能性做条件平均。

再由式 (6.4) 换算

$$
s_\theta(t,x)
\mathrel{=}
-\frac{\eta_\theta(t,x)}{\gamma(t)}.
$$

也可以直接回归 score 标签：

$$
\mathbb E
\left\|
s_\theta(t,x_t)+\frac{z}{\gamma(t)}
\right\|^2.
$$

但当 $t$ 接近端点时，$\gamma(t)\to0$，标签 $z/\gamma(t)$ 可能很大。

> **备注：实际训练中的端点不稳定处理**
>
> 常用方法包括：
>
> 1. 直接预测有界得多的 denoiser 或 noise；
> 2. 不在精确端点采样时间；
> 3. 对不同时间使用合适的 loss weighting；
> 4. 对极端时间或极端系数进行截断。
>
> 本文只需要知道：理论上的 score 与实际网络参数化可以不同，但它们包含等价信息。

### 8.3 一次基本训练迭代

1. 采样时间 $t$；
2. 采样端点 $(x_0,x_1)\sim\nu$；
3. 采样 $z\sim\mathcal N(0,\mathrm{Id})$；
4. 构造 $x_t=I(t,x_0,x_1)+\gamma(t)z$；
5. 构造速度标签 $\partial_tI+\dot\gamma z$；
6. 回归速度，或同时回归 denoiser。

训练不需要显式计算中间密度 $\rho(t,x)$，也不需要手工计算条件期望。

---

# 第三部分：从可学习的场得到生成动力学

## 9. 概率流 ODE

构造 ODE

$$
\boxed{
\frac{\mathrm dX_t}{\mathrm dt}
\mathrel{=}
b(t,X_t),
\qquad
X_0\sim\rho_0.
}
\tag{9.1}
$$

记 $X_t$ 的密度为

$$
X_t\sim q(t,\cdot).
$$

沿用式 (5.1)--(5.5) 的测试函数论证，只需把随机插值速度换成 ODE 速度 $b$，就可得

$$
\boxed{
\partial_tq+\nabla\cdot(qb)=0.
}
\tag{9.2}
$$

由式 (5.5) 可知，随机插值密度满足

$$
\partial_t\rho+\nabla\cdot(\rho b)=0,
\qquad
\rho(0)=\rho_0,
$$

ODE 密度满足

$$
\partial_tq+\nabla\cdot(qb)=0,
\qquad
q(0)=\rho_0.
$$

比较式 (9.2) 与式 (5.5) 可知，$q$ 和 $\rho$ 满足同一个 PDE 及同一初值。在连续性方程初值问题具有唯一解时，

$$
\boxed{
q(t,x)=\rho(t,x).
}
\tag{9.3}
$$


特别地，

$$
X_1\sim\rho_1.
$$

因此，训练时虽然借助了数据端点 $x_1$，生成时只需要从 $X_0\sim\rho_0$ 出发运行学习到的 ODE。

---

## 10. 同一条边际路径的 SDE

这里必须先加一条逻辑限定：双端点插值直接给出的是 $\rho(t)$、$b(t,x)$ 和概率流 ODE，并没有产生每个小时间步的布朗增量。下面的 SDE 是在已知 $\rho,b,s$ 后，通过额外选择扩散强度 $\epsilon(t)$ 和布朗运动 $W_t$ 构造出来的另一种同边际动力学。

如果对整条双端点插值轨迹固定同一个 $z$，它通常是可微曲线；而SDE轨迹几乎处处不可微。因此二者不可能是同一个路径过程，只可能在每个固定时刻拥有相同边际。

### 10.1 Fokker--Planck 方程

考虑

$$
\mathrm dZ_t
\mathrel{=}
a(t,Z_t)\,\mathrm dt
\mathbin{+}
g(t)\,\mathrm dW_t,
$$

其中 $W_t$ 是标准布朗运动，$g(t)$ 是瞬时噪声幅度。

> **备注：SDE 中 $\mathrm dW_t$ 的直观理解**
>
> $\mathrm dW_t$ 不是普通可微函数的微分。把时间离散成一个很小的步长 $\Delta t$，布朗增量满足
>
> $$
> \Delta W_t
> \mathrel{=}
> W_{t+\Delta t}-W_t
> \sim
> \mathcal N(0,\Delta t\,\mathrm{Id}).
> $$
>
> 若 $\xi\sim\mathcal N(0,\mathrm{Id})$，那么
>
> $$
> \sqrt{\Delta t}\,\xi
> \sim
> \mathcal N(0,\Delta t\,\mathrm{Id}),
> $$
>
> 所以在分布意义下
>
> $$
> \boxed{
> \Delta W_t
> \overset{\mathrm d}{=}
> \sqrt{\Delta t}\,\xi.
> }
> $$
>
> 因而 SDE 的一个 Euler 小步可以读成
>
> $$
> Z_{t+\Delta t}
> \approx
> Z_t
> \mathbin{+}
> a(t,Z_t)\Delta t
> \mathbin{+}
> g(t)\sqrt{\Delta t}\,\xi.
> $$
>
> 漂移位移是 $\Delta t$ 量级，随机位移是 $\sqrt{\Delta t}$ 量级；并且每一步都要重新采样独立的 $\xi$。这正是它与整条插值共用同一个 $z$ 的根本区别。

若 $Z_t$ 的密度为 $p(t,x)$，则

$$
\boxed{
\partial_tp
\mathrel{=}
-\nabla\cdot(ap)
\mathbin{+}
\frac12g^2(t)\Delta p.
}
\tag{10.1}
$$

> **备注：从 SDE 小步得到 Fokker--Planck 方程**
>
> 先看一维，并暂时把 $a=a(t,Z_t)$、$g=g(t)$ 写短。小步增量为
>
> $$
> \Delta Z
> \mathrel{=}
> a\,\Delta t+g\sqrt{\Delta t}\,\xi,
> \qquad
> \mathbb E[\xi]=0,\quad
> \mathbb E[\xi^2]=1.
> $$
>
> 对光滑测试函数 $\varphi$ 做二阶 Taylor 展开：
>
> $$
> \varphi(Z_t+\Delta Z)
> \mathrel{=}
> \varphi(Z_t)
> \mathbin{+}
> \varphi'(Z_t)\Delta Z
> \mathbin{+}
> \frac12\varphi''(Z_t)(\Delta Z)^2
> \mathbin{+}
> \text{更高阶项}.
> $$
>
> 条件于当前 $Z_t$ 取平均，有
>
> $$
> \mathbb E[\Delta Z\mid Z_t]
> \mathrel{=}
> a\,\Delta t,
> $$
>
> 而
>
> $$
> \mathbb E[(\Delta Z)^2\mid Z_t]
> \mathrel{=}
> g^2\Delta t+O((\Delta t)^2).
> $$
>
> 注意：随机增量本身是 $\sqrt{\Delta t}$ 量级，但平方后正好是 $\Delta t$ 量级，所以二阶项在连续极限中不会消失。除以 $\Delta t$ 并令 $\Delta t\to0$，得到
>
> $$
> \frac{\mathrm d}{\mathrm dt}\mathbb E[\varphi(Z_t)]
> \mathrel{=}
> \mathbb E\!\left[
> a\,\varphi'(Z_t)
> \mathbin{+}
> \frac12g^2\varphi''(Z_t)
> \right].
> $$
>
> 再用密度 $p$ 写成积分，并分别做一次和两次分部积分：
>
> $$
> \begin{aligned}
> \int a\varphi' p\,\mathrm dx
> &=
> -\int\varphi\,\partial_x(ap)\,\mathrm dx,
> \\
> \frac12g^2\int\varphi''p\,\mathrm dx
> &=
> \frac12g^2\int\varphi\,\partial_x^2p\,\mathrm dx.
> \end{aligned}
> $$
>
> 因为 $\varphi$ 任意，
>
> $$
> \partial_tp
> \mathrel{=}
> -\partial_x(ap)
> \mathbin{+}
> \frac12g^2\partial_x^2p.
> $$
>
> 推广到 $d$ 维后，一阶空间导数变成散度，二阶导数之和变成 $\Delta$，于是得到式 (10.1)。

由式 (10.1) 可知，令

$$
\epsilon(t):=\frac12g^2(t),
\qquad
g(t)=\sqrt{2\epsilon(t)}.
$$

则

$$
\partial_tp
\mathrel{=}
-\nabla\cdot(ap)
\mathbin{+}
\epsilon(t)\Delta p.
$$

这里定义 $\epsilon=g^2/2$ 只是为了把 Fokker--Planck 中的扩散系数写得更简洁。相应地，SDE 的噪声幅度必须写成 $g=\sqrt{2\epsilon}$；若误写成 $\sqrt{\epsilon}$，Fokker--Planck 中只会得到 $\tfrac12\epsilon\Delta p$。

> **备注：Laplace 项的来源**
>
> 在很小时间步 $\Delta t$ 内，布朗增量的标准差是 $\sqrt{\Delta t}$。二阶 Taylor 展开中的平方项因此是 $\Delta t$ 量级，极限中不会消失，最终产生 $\Delta p$。

### 10.2 只给 ODE 加噪声为什么不行

由式 (5.5) 可知，概率流 ODE 的目标边际密度已经满足连续性方程

$$
\partial_t\rho
\mathrel{=}
-\nabla\cdot(b\rho).
$$

现在保留原速度场 $b$ 作为漂移，并直接加入布朗噪声：

$$
\mathrm dZ_t
\mathrel{=}
b(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\epsilon(t)}\,\mathrm dW_t,
$$

记这个 SDE 的实际密度为 $p(t,x)$。由式 (10.1) 可知，它满足 Fokker--Planck 方程

$$
\partial_t p
\mathrel{=}
-\nabla\cdot(bp)
\mathbin{+}
\epsilon(t)\Delta p.
$$

如果希望它仍然沿着目标边际 $\rho(t,x)$ 演化，就应当能令 $p=\rho$。但把候选密度 $\rho$ 代入上式得到

$$
\partial_t\rho
\mathrel{=}
-\nabla\cdot(b\rho)
\mathbin{+}
\epsilon(t)\Delta\rho,
$$

这比 $\rho$ 原本满足的连续性方程多出了

$$
\epsilon(t)\Delta\rho.
$$


它沿密度下降方向把概率从高密度区域摊向低密度区域，因此会改变原来的边际路径。

### 10.3 用 score 补偿扩散

由式 (6.1) 的 score 定义可知

$$
s=\nabla\log\rho
\mathrel{=}
\frac{\nabla\rho}{\rho},
$$

所以

$$
\rho s=\nabla\rho.
$$

选择前向漂移

$$
\boxed{
b_F(t,x)
\mathrel{=}
b(t,x)+\epsilon(t)s(t,x).
}
\tag{10.2}
$$

构造前向 SDE

$$
\boxed{
\mathrm dZ_t
\mathrel{=}
\bigl(b+\epsilon s\bigr)(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\epsilon(t)}\,\mathrm dW_t,
\qquad
Z_0\sim\rho_0.
}
\tag{10.3}
$$

将式 (10.2) 和式 (10.3) 代入式 (10.1) 的 Fokker--Planck 方程，并令候选密度为 $\rho$，得到

$$
\begin{aligned}
\partial_t\rho
&=
-\nabla\cdot\bigl((b+\epsilon s)\rho\bigr)
\mathbin{+}
\epsilon\Delta\rho
\\
&=
-\nabla\cdot(b\rho)
-\epsilon\nabla\cdot(\rho s)
\mathbin{+}
\epsilon\Delta\rho
\\
&=
-\nabla\cdot(b\rho)
-\epsilon\Delta\rho
\mathbin{+}
\epsilon\Delta\rho
\\
&=
-\nabla\cdot(b\rho).
\end{aligned}
$$

> **备注：两个 $\Delta\rho$ 抵消的直观理解**
>
> 布朗扩散产生
>
> $$
> +\epsilon\Delta\rho,
> $$
>
> 它把概率从高密度区域向周围摊开。新增的 score 漂移具有概率流量
>
> $$
> \rho(\epsilon s)
> \mathrel{=}
> \epsilon\rho\nabla\log\rho
> \mathrel{=}
> \epsilon\nabla\rho.
> $$
>
> 把它放入连续性方程，会贡献
>
> $$
> -\nabla\cdot(\epsilon\nabla\rho)
> \mathrel{=}
> -\epsilon\Delta\rho.
> $$
>
> 因此 score 漂移恰好补偿布朗扩散对单时刻密度的摊平作用。注意，它们抵消的是密度方程中的净效应，不是逐条样本轨迹上的随机波动；SDE 轨迹仍然是随机的。

最后一行与式 (5.5) 的目标连续性方程一致。

在相应 PDE 解唯一时，

$$
\boxed{
p(t,x)=\rho(t,x).
}
\tag{10.4}
$$

### 10.4 前向、反向与确定性动力学

汇总式 (9.1) 的概率流 ODE、式 (10.3) 的前向 SDE 以及相应的反向 SDE，同一条边际路径对应三种常用动力学：

$$
\boxed{
\begin{aligned}
\text{概率流 ODE：}\quad
&\dot X_t=b(t,X_t),
\\
\text{前向 SDE：}\quad
&\mathrm dZ_t=(b+\epsilon s)\,\mathrm dt
+\sqrt{2\epsilon}\,\mathrm dW_t,
\\
\text{反向 SDE：}\quad
&\mathrm dZ_t=(b-\epsilon s)\,\mathrm dt
+\sqrt{2\epsilon}\,\mathrm d\overline W_t,
\quad t:1\to0.
\end{aligned}}
\tag{10.5}
$$

> **备注：反向 SDE 漂移为 $b-\epsilon s$ 的原因**
>
> 反向 SDE 沿 $t$ 递减运行。设实际走了一小段长度 $h>0$：
>
> $$
> t\longrightarrow t-h,
> \qquad
> \Delta t=-h.
> $$
>
> SDE 的一步更新是
>
> $$
> Z_{t-h}
> \mathrel{=}
> Z_t
> -u_R(t,Z_t)h
> +\sqrt{2\epsilon h}\,\xi,
> \qquad
> \xi\sim\mathcal N(0,\mathrm{Id}),
> $$
>
> 其中在这一小步内记 $\epsilon=\epsilon(t)$。注意：由于 $\Delta t=-h$，漂移造成的实际位移是 $-u_Rh$；布朗增量的方差仍为 $2\epsilon h$，不会因为时间反向而变成负数。
>
> 下面只比较这一小步造成的密度变化。
>
> 如果概率质量沿速度 $c$ 正向移动长度 $h$，连续性方程告诉我们，密度变化为
>
> $$
> -h\,\nabla\cdot(\rho c).
> $$
>
> 当前反向一步的实际漂移速度是 $-u_R$，所以漂移造成的密度变化为
>
> $$
> +h\,\nabla\cdot(\rho u_R).
> $$
>
> 同时，布朗噪声会把密度摊开。由 10.2 节，它在长度 $h$ 内额外产生
>
> $$
> +\epsilon h\,\Delta\rho
> \mathrel{=}
> +h\,\nabla\cdot(\epsilon\nabla\rho).
> $$
>
> 因此，反向 SDE 的一步总密度变化是
>
> $$
> h\,\nabla\cdot
> \bigl(\rho u_R+\epsilon\nabla\rho\bigr).
> $$
>
> 现在取
>
> $$
> \boxed{u_R=b-\epsilon s},
> $$
>
> 并把它代入上面的一步总密度变化：
>
> $$
> \begin{aligned}
> h\,\nabla\cdot
> \bigl(\rho u_R+\epsilon\nabla\rho\bigr)
> &=
> h\,\nabla\cdot
> \bigl(\rho(b-\epsilon s)+\epsilon\nabla\rho\bigr)
> \\
> &=
> h\,\nabla\cdot
> \bigl(\rho b-\epsilon\rho s+\epsilon\nabla\rho\bigr)
> \\
> &=
> h\,\nabla\cdot
> \bigl(\rho b-\epsilon\nabla\rho+\epsilon\nabla\rho\bigr)
> \\
> &=
> h\,\nabla\cdot(\rho b).
> \end{aligned}
> $$
>
> 这里使用了 $\rho s=\nabla\rho$。最后得到的 $h\,\nabla\cdot(\rho b)$，正是目标 ODE 从 $t$ 倒走到 $t-h$ 时应有的密度变化。因此，反向 SDE 与目标 ODE 沿着同一条边际路径反向演化。
>
> 最后要区分“公式中的漂移”与“实际一步的位移”。虽然反向 SDE 写的是 $u_R=b-\epsilon s$，但因为 $\Delta t=-h$，其确定性位移实际为
>
> $$
> -u_Rh
> \mathrel{=}
> -bh+\epsilon s\,h.
> $$
>
> 其中 $-bh$ 反转原来的概率输运，$+\epsilon s\,h$ 沿密度升高方向拉回样本，恰好补偿这一步中新加入的布朗噪声所造成的摊平。

> **备注：ODE 可直接倒着积分而 SDE 需要 score 修正的原因**
>
> ODE 的轨迹由速度场唯一决定，把时间步改成负数就会沿原轨迹返回。
>
> SDE 每一小步都加入新噪声。反向过程不仅要反转平均移动，还要修正扩散造成的概率摊平；这个修正由 score 提供。

![图 3：$\epsilon$ 对 ODE/SDE 样本轨迹的影响](images/paper-figure-05-epsilon-trajectories.png)

*说明：$\epsilon=0$ 是确定性 ODE；$\epsilon>0$ 后，同一边际路径可以由越来越随机的 SDE 轨迹实现。图中单条轨迹明显不同，但理想场下每个时刻的总体密度不依赖 $\epsilon$。来源：[论文 v4, Figure 5](https://arxiv.org/abs/2303.08797v4)。*

### 10.5 把 $b=v-\gamma\dot\gamma s$ 代入

由式 (7.2) 可知

$$
b=v-\gamma\dot\gamma s.
$$

将上式分别代入式 (10.2) 的前向漂移和式 (10.5) 的反向漂移，可知

$$
\boxed{
b_F
\mathrel{=}
v+\bigl(\epsilon-\gamma\dot\gamma\bigr)s,
}
\tag{10.6}
$$

$$
\boxed{
b_B
\mathrel{=}
v-\bigl(\epsilon+\gamma\dot\gamma\bigr)s.
}
\tag{10.7}
$$

这两个公式同时包含：

- 确定性插值速度 $v$；
- latent noise schedule $\gamma$；
- 生成 SDE diffusion schedule $\epsilon$；
- 边际 score $s$。

---

## 11. 理想选择模型

选择 ODE 还是 SDE，取决于目标：

| 目标 | 常见选择 |
|---|---|
| 给定初始噪声后希望结果确定 | ODE |
| 希望保留随机采样轨迹 | SDE |
| 希望使用成熟的 ODE 求解器 | ODE |
| 更重视扩散带来的误差鲁棒性 | SDE |


---

# 第四部分：从统一框架推出常见算法

这一部分不再引入新的统一原理，只把前面已经得到的公式具体化。

主线是

$$
\boxed{
\text{选择特殊的 }I,\gamma,\rho_0
\quad\Longrightarrow\quad
\text{得到 Flow Matching 或 diffusion 的常用公式。}
}
$$

---

## 12. Endpoint Flow Matching：令 $\gamma=0$

在式 (2.1) 中取

$$
\gamma(t)=0,
$$

由式 (2.1) 和式 (4.1) 可知

$$
x_t=I(t,x_0,x_1),
$$

随机速度标签变为

$$
Y_t=\partial_tI(t,x_0,x_1).
$$

再由式 (4.2) 可知，条件平均速度是

$$
\boxed{
b(t,x)
\mathrel{=}
\mathbb E[
\partial_tI(t,x_0,x_1)
\mid
I(t,x_0,x_1)=x
].
}
\tag{12.1}
$$

将上述路径与速度标签代入式 (8.1)，得到训练目标

$$
\boxed{
\mathcal L_{\mathrm{FM}}(\theta)
\mathrel{=}
\mathbb E
\left\|
b_\theta\bigl(t,I(t,x_0,x_1)\bigr)
\mathbin{-}
\partial_tI(t,x_0,x_1)
\right\|^2.
}
\tag{12.2}
$$

训练完成后运行

$$
\frac{\mathrm dX_t}{\mathrm dt}
\mathrel{=}
b_\theta(t,X_t),
\qquad
X_0\sim\rho_0.
$$

这就是本文所说的 endpoint Flow Matching。

> **备注：Flow Matching 与 ODE 不是同一个概念。**
>
> Flow Matching 是学习速度场的方法；ODE 是训练完成后使用该速度场生成样本的动力学。

---

## 13. Linear Flow Matching 与 Rectified Flow

### 13.1 线性端点插值

取

$$
I(t,x_0,x_1)
\mathrel{=}
(1-t)x_0+tx_1.
$$

则

$$
\partial_tI=x_1-x_0.
$$

将该线性插值及 $\partial_tI=x_1-x_0$ 代入式 (12.2)，得到

$$
\boxed{
\mathcal L_{\mathrm{linear\text{-}FM}}(\theta)
\mathrel{=}
\mathbb E
\left\|
b_\theta\bigl(t,(1-t)x_0+tx_1\bigr)
\mathbin{-}
(x_1-x_0)
\right\|^2.
}
\tag{13.1}
$$

这也是 Rectified Flow 的首轮基础训练形式。

虽然训练样本的插值线是直线，但生成 ODE 的轨迹不一定是直线。原因是网络学到的是所有经过同一 $(t,x)$ 的随机标签的条件平均。

### 13.2 reflow 的基本思想

Rectified Flow 的特色不只是使用线性插值，还包括重新配对：

1. 先用独立端点对训练一个速度场；
2. 从 $x_0\sim\rho_0$ 出发运行当前 ODE，得到生成终点 $\widehat x_1$；
3. 把同一条生成轨迹的 $(x_0,\widehat x_1)$ 重新配成端点对；
4. 在这些新端点对之间再次使用线性插值训练。

新耦合比完全独立的噪声—数据配对更接近当前生成映射，因此重新训练后的轨迹往往更直、更容易用少量 ODE 步数逼近。


---

## 14. One-sided Gaussian 随机插值

当基分布是标准高斯时，可以直接写成

$$
\boxed{
x_t
\mathrel{=}
\alpha(t)z+\beta(t)x_1,
\qquad
z\sim\mathcal N(0,\mathrm{Id}),
\quad
x_1\sim\rho_1.
}
\tag{14.1}
$$

要求

$$
\alpha(0)=1,
\qquad
\beta(0)=0,
$$

$$
\alpha(1)=0,
\qquad
\beta(1)=1.
$$

于是

$$
x_0=z\sim\mathcal N(0,\mathrm{Id}),
\qquad
x_1\sim\rho_1.
$$

与式 (2.1) 比较可知，它是一般双端点框架的线性特例：

$$
I(t,x_0,x_1)
\mathrel{=}
\alpha(t)x_0+\beta(t)x_1,
\qquad
\gamma(t)=0,
$$

其中 $x_0=z$。

对式 (14.1) 关于时间求导，得到随机速度标签

$$
Y_t
\mathrel{=}
\dot\alpha(t)z+\dot\beta(t)x_1.
$$

由式 (4.2) 对随机速度做条件平均，得到

$$
\boxed{
b(t,x)
\mathrel{=}
\dot\alpha(t)\mathbb E[z\mid x_t=x]
\mathbin{+}
\dot\beta(t)\mathbb E[x_1\mid x_t=x].
}
\tag{14.2}
$$

由于式 (14.1) 在给定 $x_1$ 时是噪声尺度为 $\alpha(t)$ 的条件高斯，沿用式 (6.2) 的边际 score 恒等式可知

$$
\boxed{
s(t,x)
\mathrel{=}
-\frac1{\alpha(t)}
\mathbb E[z\mid x_t=x].
}
\tag{14.3}
$$

对式 (14.1) 在条件 $x_t=x$ 下取期望，还得到恒等式

$$
x
\mathrel{=}
\alpha(t)\mathbb E[z\mid x_t=x]
\mathbin{+}
\beta(t)\mathbb E[x_1\mid x_t=x].
\tag{14.4}
$$

因此平均噪声、平均数据端点、score 和速度可以相互转换。

这正是从统一随机插值走向常见 diffusion 参数化的桥梁。

---

## 15. 切换到 diffusion 的前向加噪时间

前面一直使用生成时间

$$
t: \text{噪声}\to\text{数据}.
$$

扩散模型文献更常使用加噪时间

$$
\tau:0\longrightarrow T,
\qquad
\text{数据}\to\text{噪声}.
$$

二者方向相反。为避免混淆，本节改用 $y_\tau$：

若生成时间为 $t\in[0,1]$，扩散时间为 $\tau\in[0,T]$，可以记成

$$
\tau=T(1-t).
$$

当 $T=1$ 时，它就是 $\tau=1-t$。因此，生成时间中的“起点噪声”对应加噪时间中的“终点噪声”。将式 (14.1) 按上述关系反转时间，再更换端点与系数记号，可写成

$$
\boxed{
y_\tau
\mathrel{=}
a(\tau)y_0+\sigma(\tau)\varepsilon,
}
\tag{15.1}
$$

其中

$$
y_0\sim p_{\mathrm{data}},
\qquad
\varepsilon\sim\mathcal N(0,\mathrm{Id}).
$$

端点条件是

$$
a(0)=1,\qquad \sigma(0)=0,
$$

$$
a(T)\approx0,\qquad \sigma(T)\approx1.
$$

因此，式 (15.1) 与式 (14.1) 描述的是相同的 one-sided Gaussian 固定时刻边际，但时间方向相反。


---

## 16. noise、score 与 velocity 参数化

令 $\rho(\tau,y)$ 表示 $y_\tau$ 的边际密度，并定义

$$
s_\tau(y)=\nabla_y\log\rho(\tau,y).
$$

### 16.1 noise 与 score

对式 (15.1) 的条件高斯密度使用式 (14.3) 的同一 score 恒等式，得到

$$
\boxed{
s_\tau(y)
\mathrel{=}
-\frac1{\sigma(\tau)}
\mathbb E[\varepsilon\mid y_\tau=y].
}
\tag{16.1}
$$

若噪声网络学习

$$
\varepsilon_\theta(y,\tau)
\approx
\mathbb E[\varepsilon\mid y_\tau=y],
$$

将该预测代入式 (16.1) 可知

$$
\boxed{
s_\theta(y,\tau)
\mathrel{=}
-\frac{\varepsilon_\theta(y,\tau)}{\sigma(\tau)}.
}
\tag{16.2}
$$

### 16.2 noise 与 velocity

对式 (15.1) 关于 $\tau$ 求导，可知

$$
Y_\tau
\mathrel{=}
\dot a(\tau)y_0+\dot\sigma(\tau)\varepsilon.
$$

再由式 (15.1) 可知

$$
y_0
\mathrel{=}
\frac{y-\sigma(\tau)\varepsilon}{a(\tau)},
$$

将上式代入速度表达式，得到

$$
Y_\tau
\mathrel{=}
\frac{\dot a}{a}y
\mathbin{+}
\left(
\dot\sigma-\frac{\dot a}{a}\sigma
\right)\varepsilon.
$$

定义

$$
A(\tau):=\frac{\dot a(\tau)}{a(\tau)},
$$

$$
B(\tau):=
\dot\sigma(\tau)-A(\tau)\sigma(\tau).
$$

在条件 $y_\tau=y$ 下取期望，得到

$$
\boxed{
u(\tau,y)
\mathrel{=}
A(\tau)y
\mathbin{+}
B(\tau)
\mathbb E[\varepsilon\mid y_\tau=y].
}
\tag{16.3}
$$

因此由式 (16.2) 和式 (16.3) 可知，理想模型下

$$
\boxed{
\begin{aligned}
s_\theta
&=-\frac{\varepsilon_\theta}{\sigma},
\\
u_\theta
&=Ay+B\varepsilon_\theta,
\\
\varepsilon_\theta
&=\frac{u_\theta-Ay}{B}.
\end{aligned}}
\tag{16.4}
$$

> **例子：一维线性加噪中三种参数化如何换算？**
>
> 取
>
> $$
> a(\tau)=1-\tau,
> \qquad
> \sigma(\tau)=\tau,
> $$
>
> 则
>
> $$
> y_\tau=(1-\tau)y_0+\tau\varepsilon,
> \qquad
> Y_\tau=\varepsilon-y_0.
> $$
>
> 令 $\tau=\tfrac12$，观察到 $y_\tau=1$，并假设网络给出的条件平均噪声为
>
> $$
> \mathbb E[\varepsilon\mid y_{1/2}=1]=0.2.
> $$
>
> 由 $1=\tfrac12y_0+\tfrac12\varepsilon$，
>
> $$
> \mathbb E[y_0\mid y_{1/2}=1]
> \mathrel{=}
> \frac{1-\frac12\times0.2}{1/2}
> \mathrel{=}
> 1.8.
> $$
>
> 所以条件平均速度为
>
> $$
> u\!\left(\tfrac12,1\right)
> \mathrel{=}
> 0.2-1.8
> \mathrel{=}
> -1.6,
> $$
>
> score 为
>
> $$
> s_{1/2}(1)
> \mathrel{=}
> -\frac{0.2}{1/2}
> \mathrel{=}
> -0.4.
> $$
>
> 同一个噪声预测 $0.2$ 同时确定了速度预测 $-1.6$ 和 score 预测 $-0.4$。这就是三种参数化“信息等价”的具体含义。

所以三种参数化包含相同的条件期望信息：

$$
\boxed{
\text{noise prediction}
\Longleftrightarrow
\text{score prediction}
\Longleftrightarrow
\text{velocity prediction}.
}
$$

> **备注：转换所需的条件**
>
> 上式要求对应分母不为零，例如 $\sigma(\tau)>0$、$a(\tau)>0$、$B(\tau)\neq0$。端点附近通常要使用更稳定的参数化或单独处理。

---

## 17. VP 路径与三条连续动力学

由式 (15.1) 可知，one-sided Gaussian 路径写成

$$
y_\tau
\mathrel{=}
a(\tau)y_0+\sigma(\tau)\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,\mathrm{Id}).
$$

Variance Preserving（VP）路径进一步要求

$$
\boxed{
a^2(\tau)+\sigma^2(\tau)=1.
}
\tag{17.1}
$$

选择连续噪声率

$$
\beta(\tau)\ge 0,
$$

并定义

$$
\boxed{
a(\tau)
\mathrel{=}
\exp\left(
-\frac12\int_0^\tau\beta(r)\,\mathrm dr
\right),
\qquad
\sigma^2(\tau)=1-a^2(\tau).
}
\tag{17.2}
$$

对式 (17.2) 求导，再使用 $\sigma^2=1-a^2$，可知

$$
\boxed{
\frac{\dot a(\tau)}{a(\tau)}
\mathrel{=}
-\frac12\beta(\tau),
\qquad
\dot\sigma(\tau)
\mathrel{=}
\frac{\beta(\tau)a^2(\tau)}{2\sigma(\tau)}.
}
\tag{17.3}
$$

下面在这组 VP 单时刻边际之外，再选择线性、各向同性的 Markov 加噪动力学，依次得到前向 VP SDE、反向 VP SDE 和概率流 ODE。

### 17.1 前向 VP SDE

对应的前向加噪动力学取为

$$
\boxed{
\mathrm dy_\tau
\mathrel{=}
-\frac12\beta(\tau)y_\tau\,\mathrm d\tau
\mathbin{+}
\sqrt{\beta(\tau)}\,\mathrm dW_\tau.
}
\tag{17.4}
$$

> **备注：式 (17.4) 的直观理解**
>
> 把它看成 DDPM 前向加噪一步在步长趋近于零时的写法。设时间向前走一个很小的步长 $h>0$，并在这一步加入方差为 $\beta(\tau)h$ 的高斯噪声：
>
> $$
> y_{\tau+h}
> \mathrel{=}
> \sqrt{1-\beta(\tau)h}\,y_\tau
> \mathbin{+}
> \sqrt{\beta(\tau)h}\,\varepsilon,
> \qquad
> \varepsilon\sim\mathcal N(0,\mathrm{Id}).
> $$
>
> 旧信号的方差权重是 $1-\beta h$，新噪声的方差权重是 $\beta h$，两者之和仍为 $1$：
>
> $$
> (1-\beta h)+\beta h=1.
> $$
>
> 这就是 Variance Preserving 的含义。对信号系数作一阶展开：
>
> $$
> \sqrt{1-\beta h}
> \mathrel{=}
> 1-\frac12\beta h+O(h^2),
> $$
>
> 于是
>
> $$
> y_{\tau+h}-y_\tau
> \mathrel{=}
> -\frac12\beta(\tau)y_\tau h
> \mathbin{+}
> \sqrt{\beta(\tau)}\sqrt h\,\varepsilon
> +O(h^2).
> $$
>
> 布朗运动在长度为 $h$ 的时间段内的增量满足
>
> $$
> \Delta W_\tau
> \sim
> \mathcal N(0,h\,\mathrm{Id}),
> \qquad
> \Delta W_\tau
> \overset{\mathrm d}{=}
> \sqrt h\,\varepsilon.
> $$
>
> 把 $\sqrt h\,\varepsilon$ 记为 $\Delta W_\tau$，再令 $h\to0$，就得到式 (17.4)。其中 $\sqrt\beta\,\mathrm dW$ 不断加入新高斯噪声，$-\tfrac12\beta y\,\mathrm d\tau$ 同时缩小旧信号，为新噪声腾出方差空间。这个构造还隐含了线性、各向同性且每步使用独立高斯增量的 Markov 加噪选择；只给定单时刻 VP 边际本身，并不唯一决定整条随机路径。

记

$$
f(\tau,y)=-\frac12\beta(\tau)y,
\qquad
g^2(\tau)=\beta(\tau).
$$

式 (17.4) 是线性 SDE。由式 (17.3) 中 $\dot a/a=-\beta/2$ 可知，对式 (17.4) 使用积分因子 $a^{-1}(\tau)$ 后有

$$
\mathrm d\!\left(\frac{y_\tau}{a(\tau)}\right)
\mathrel{=}
\frac{\sqrt{\beta(\tau)}}{a(\tau)}\,\mathrm dW_\tau.
$$

从 $0$ 积分到 $\tau$，得到

$$
y_\tau
\mathrel{=}
a(\tau)y_0
\mathbin{+}
a(\tau)
\int_0^\tau
\frac{\sqrt{\beta(r)}}{a(r)}\,\mathrm dW_r.
$$

伊藤积分的条件均值为零，而由式 (17.2) 和式 (17.3) 可知，其条件方差为

$$
\begin{aligned}
a^2(\tau)
\int_0^\tau\frac{\beta(r)}{a^2(r)}\,\mathrm dr
&=
a^2(\tau)\left(a^{-2}(\tau)-1\right)
\\
&=
1-a^2(\tau)
\mathrel{=}
\sigma^2(\tau).
\end{aligned}
$$

因此，给定初始数据 $y_0$，前向 VP SDE 在任意固定时刻的条件边际为

$$
\boxed{
y_\tau\mid y_0
\overset{\mathrm d}{=}
a(\tau)y_0+\sigma(\tau)\varepsilon.
}
\tag{17.5}
$$

式 (17.5) 只说明固定时刻的条件分布相同。若所有时刻共用一个 $\varepsilon$，右边形成一条可微随机插值轨迹；真正的 VP SDE 则由布朗运动驱动，整条轨迹几乎处处不可微。二者相同的是单时刻边际，不是路径联合分布。

### 17.2 反向 VP SDE

一般前向 SDE

$$
\mathrm dy_\tau
\mathrel{=}
f(\tau,y_\tau)\,\mathrm d\tau
+g(\tau)\,\mathrm dW_\tau
$$

由式 (10.5) 可知，前向与反向漂移分别为 $b+\epsilon s$ 和 $b-\epsilon s$。消去概率流速度 $b$，再使用 $g^2=2\epsilon$，可知一般前向 SDE 对应的反向 SDE 漂移为

$$
f-g^2s,
\qquad
s_\tau(y)=\nabla_y\log\rho(\tau,y).
$$

将式 (17.4) 中的 $f=-\beta y/2$ 和 $g^2=\beta$ 代入上式，得到 VP 的反向 SDE

$$
\boxed{
\mathrm dy_\tau
\mathrel{=}
\left[
-\frac12\beta(\tau)y_\tau
-\beta(\tau)s_\tau(y_\tau)
\right]\mathrm d\tau
\mathbin{+}
\sqrt{\beta(\tau)}\,\mathrm d\overline W_\tau,
\qquad
\tau:T\to0.
}
\tag{17.6}
$$

由式 (16.2) 的 noise--score 关系

$$
s_\theta(y,\tau)
\mathrel{=}
-\frac{\varepsilon_\theta(y,\tau)}{\sigma(\tau)},
$$

将其代入式 (17.6)，得到噪声预测形式

$$
\boxed{
\mathrm dy_\tau
\mathrel{=}
\left[
-\frac12\beta(\tau)y_\tau
\mathbin{+}
\beta(\tau)
\frac{\varepsilon_\theta(y_\tau,\tau)}{\sigma(\tau)}
\right]\mathrm d\tau
\mathbin{+}
\sqrt{\beta(\tau)}\,\mathrm d\overline W_\tau,
\qquad
\tau:T\to0.
}
\tag{17.7}
$$

这里公式中的 $\mathrm d\tau$ 随生成方向为负。若实际从 $\tau$ 走到 $\tau-h$，其中 $h>0$，则确定性位移等于漂移乘以 $-h$，布朗增量的方差仍然是正的 $\beta(\tau)h$。

### 17.3 VP 概率流 ODE

由式 (10.2) 中 $b_F=b+\epsilon s$ 可知，对一般前向 SDE 的漂移 $f=b_F$ 和扩散强度 $g^2=2\epsilon$，与它共享边际的概率流 ODE 速度为

$$
b=f-\frac12g^2s.
$$

将式 (17.4) 中 VP 的 $f$ 和 $g$ 代入上式，得到

$$
\boxed{
\frac{\mathrm dy_\tau}{\mathrm d\tau}
\mathrel{=}
-\frac12\beta(\tau)y_\tau
-\frac12\beta(\tau)s_\tau(y_\tau).
}
\tag{17.8}
$$

再将式 (16.2) 的 $s_\theta=-\varepsilon_\theta/\sigma$ 代入式 (17.8)，得到

$$
\boxed{
\frac{\mathrm dy_\tau}{\mathrm d\tau}
\mathrel{=}
-\frac12\beta(\tau)y_\tau
\mathbin{+}
\frac12\beta(\tau)
\frac{\varepsilon_\theta(y_\tau,\tau)}{\sigma(\tau)}.
}
\tag{17.9}
$$

生成时从 $y_T$ 出发，沿 $\tau:T\to0$ 反向积分。概率流 ODE 不注入新的布朗噪声，所以给定相同的 $y_T$、模型和求解设置后，连续轨迹是确定的。

### 17.4 三条连续动力学并排比较

汇总式 (17.4)、式 (17.7) 和式 (17.9)：

$$
\boxed{
\begin{aligned}
\text{前向 VP SDE：}\quad
\mathrm dy_\tau
&=
-\frac12\beta y_\tau\,\mathrm d\tau
\mathbin{+}
\sqrt\beta\,\mathrm dW_\tau,
\\[1mm]
\text{反向 VP SDE：}\quad
\mathrm dy_\tau
&=
\left[
-\frac12\beta y_\tau
\mathbin{+}
\beta\frac{\varepsilon_\theta}{\sigma}
\right]\mathrm d\tau
\mathbin{+}
\sqrt\beta\,\mathrm d\overline W_\tau,
\quad \tau:T\to0,
\\[1mm]
\text{VP 概率流 ODE：}\quad
\frac{\mathrm dy_\tau}{\mathrm d\tau}
&=
-\frac12\beta y_\tau
\mathbin{+}
\frac12\beta\frac{\varepsilon_\theta}{\sigma}.
\end{aligned}
}
\tag{17.10}
$$

其中 $\beta$、$\sigma$ 和网络输入的时间依赖暂时省略，以突出结构。

$$
\boxed{
\begin{array}{c|c|c}
&\varepsilon_\theta/\sigma\text{ 的漂移系数}&\text{是否注入新随机噪声}\\
\hline
\text{反向 VP SDE}&\beta&\text{是}\\
\text{VP 概率流 ODE}&\beta/2&\text{否}
\end{array}
}
\tag{17.11}
$$

由式 (17.10) 和式 (17.11) 可知，在连续时间层面，二者的关系为

$$
\boxed{
\text{反向 VP SDE 与概率流 ODE 的 score/noise 漂移只差一个 }\frac12\text{，}
\text{同时前者还有布朗噪声。}
}
\tag{17.12}
$$

但连续方程还不是有限步算法。DDPM 和 DDIM 的标准更新公式，还取决于怎样把连续时间切成有限步，以及每一步采用什么跨步规则。

---

## 18. 什么是“离散方式”

连续 ODE

$$
\frac{\mathrm dy_\tau}{\mathrm d\tau}
\mathrel{=}
v(y_\tau,\tau)
$$

只规定每一个瞬间的速度。理论上的一步满足

$$
y_{\tau_{k-1}}
\mathrel{=}
y_{\tau_k}
\mathbin{+}
\int_{\tau_k}^{\tau_{k-1}}
v(y_\tau,\tau)\,\mathrm d\tau.
\tag{18.1}
$$

但积分中的整段轨迹 $y_\tau$ 尚未知道。计算机只能选取有限个时间点

$$
\tau_K>\tau_{K-1}>\cdots>\tau_0,
$$

并规定怎样从 $y_k$ 近似计算 $y_{k-1}$。这个有限步跨越规则就是离散方式或数值求解方法。

### 18.1 反向普通 Euler

定义正的反向步长

$$
\boxed{
h_k:=\tau_k-\tau_{k-1}>0,
\qquad
\tau_{k-1}=\tau_k-h_k.
}
\tag{18.2}
$$

由式 (18.1) 可知，普通 Euler 在整一步内用起点速度近似积分中的整段速度：

$$
\boxed{
y_{k-1}^{\mathrm{Euler}}
\mathrel{=}
y_k-h_kv(y_k,\tau_k).
}
\tag{18.3}
$$

它的直观含义是：在当前位置读取一次切线方向，然后沿这条切线走完整一步。

### 18.2 显式中点法

中点法不能只写最终公式，还必须定义中点怎样得到。完整的反向显式中点法是

$$
\boxed{
\begin{aligned}
k_1
&=
v(y_k,\tau_k),
\\
y_{\mathrm{mid}}
&=
y_k-\frac{h_k}{2}k_1,
\\
k_2
&=
v\left(y_{\mathrm{mid}},\tau_k-\frac{h_k}{2}\right),
\\
y_{k-1}^{\mathrm{midpoint}}
&=
y_k-h_kk_2.
\end{aligned}
}
\tag{18.4}
$$

这里的 $y_{\mathrm{mid}}$ 是由起点速度预测出的中点，并不是未知端点的简单平均 $(y_k+y_{k-1})/2$。

Euler、中点法、Heun 和 Runge--Kutta 即使求解同一个 ODE，有限步轨迹通常也不同；当最大步长趋近于零时，它们才收敛到同一个连续解。

### 18.3 SDE 的 Euler--Maruyama 离散

对于反向运行的 SDE

$$
\mathrm dy_\tau
\mathrel{=}
u(y_\tau,\tau)\,\mathrm d\tau
+g(\tau)\,\mathrm d\overline W_\tau,
$$

由式 (18.2) 可知反向时间增量为 $-h_k$，而布朗增量的方差为 $h_k\mathrm{Id}$。因此 Euler--Maruyama 一步为

$$
\boxed{
y_{k-1}
\approx
y_k-h_ku(y_k,\tau_k)
+g(\tau_k)\sqrt{h_k}\,z_k,
\qquad
z_k\sim\mathcal N(0,\mathrm{Id}).
}
\tag{18.5}
$$

因此 SDE 离散不仅要规定怎样近似漂移积分，还要规定每一步的随机增量及其时间相关性。

### 18.4 “离散方式”的宏观含义

可以把它概括为：

$$
\boxed{
\text{离散方式}
\mathrel{=}
\text{在有限时间步内，规定哪些量冻结、哪些量变化，以及是否加入新随机性。}
}
\tag{18.6}
$$

普通 Euler 冻结整个速度；显式中点法在半步后重新计算速度；DDPM 保留离散高斯后验；DDIM 冻结本步估计的 $\widehat y_0$ 和 $\widehat\varepsilon$，同时让已知的 schedule 系数变化。

---

## 19. DDPM 与 DDIM 使用的离散方式

这一节按同一个顺序讨论两种算法：

$$
\boxed{
\text{连续动力学}
\longrightarrow
\text{选择有限步结构}
\longrightarrow
\text{标准更新公式}
\longrightarrow
\text{一阶展开验证连续极限}.
}
\tag{19.1}
$$

两条对应关系是

$$
\boxed{
\begin{aligned}
\text{反向 VP SDE}
&\longrightarrow
\text{离散高斯后验}
\longrightarrow
\text{DDPM},
\\
\text{VP 概率流 ODE}
&\longrightarrow
\text{本步固定信号/噪声估计并按 schedule 重组}
\longrightarrow
\text{DDIM}.
\end{aligned}
}
\tag{19.2}
$$

定义离散 schedule

$$
\alpha_k=1-\beta_k,
\qquad
\bar\alpha_k=\prod_{j=1}^k\alpha_j,
$$

以及

$$
a_k=\sqrt{\bar\alpha_k},
\qquad
\sigma_k=\sqrt{1-\bar\alpha_k}.
\tag{19.3}
$$

连续噪声率 $\beta(\tau)$ 与离散噪声量 $\beta_k$ 不是同一个量。由式 (17.2) 可知，一个时间区间内信号方差的保留比例为

$$
\alpha_k
\mathrel{=}
\exp\left(
-\int_{\tau_{k-1}}^{\tau_k}\beta(r)\,\mathrm dr
\right),
\qquad
\beta_k=1-\alpha_k.
$$

对这个指数式作小步长展开，得到

$$
\boxed{
h_k:=\tau_k-\tau_{k-1}>0,
\qquad
\beta_k=\beta(\tau_k)h_k+O(h_k^2).
}
\tag{19.4}
$$

### 19.1 DDPM 使用的离散方式：从高斯后验得到标准更新

DDPM 先定义离散前向链

$$
\boxed{
q(y_k\mid y_{k-1})
\mathrel{=}
\mathcal N
\left(
\sqrt{\alpha_k}y_{k-1},
\beta_k\mathrm{Id}
\right).
}
\tag{19.5}
$$

等价地，

$$
y_k
\mathrel{=}
\sqrt{\alpha_k}y_{k-1}
\mathbin{+}
\sqrt{\beta_k}\varepsilon_k,
\qquad
\varepsilon_k\overset{\mathrm{i.i.d.}}{\sim}\mathcal N(0,\mathrm{Id}).
$$

每一步都使用新的独立高斯增量，所以这是一条 Markov 随机链。反复代入式 (19.5)，信号系数累乘为 $\sqrt{\bar\alpha_k}$，独立噪声的累积方差为 $1-\bar\alpha_k$。再由式 (19.3) 可知，累积 $k$ 步后的固定时刻边际为

$$
\boxed{
q(y_k\mid y_0)
\mathrel{=}
\mathcal N
\left(
a_ky_0,
\sigma_k^2\mathrm{Id}
\right),
}
\tag{19.6}
$$

也就是

$$
y_k=a_ky_0+\sigma_k\varepsilon.
$$

由式 (19.5) 的一步转移和式 (19.6) 的累积边际可知，在 $y_0$ 已知时，高斯配方给出精确的离散后验

$$
q(y_{k-1}\mid y_k,y_0)
\mathrel{=}
\mathcal N
\left(
\widetilde\mu_k(y_k,y_0),
\widetilde\beta_k\mathrm{Id}
\right),
\tag{19.7}
$$

其中

$$
\widetilde\beta_k
\mathrel{=}
\frac{1-\bar\alpha_{k-1}}
{1-\bar\alpha_k}\beta_k,
$$

$$
\widetilde\mu_k(y_k,y_0)
\mathrel{=}
\frac{\sqrt{\bar\alpha_{k-1}}\beta_k}
{1-\bar\alpha_k}y_0
\mathbin{+}
\frac{\sqrt{\alpha_k}(1-\bar\alpha_{k-1})}
{1-\bar\alpha_k}y_k.
\tag{19.8}
$$

生成时不知道真实 $y_0$。网络先预测

$$
\widehat\varepsilon_k
\mathrel{=}
\varepsilon_\theta(y_k,k),
$$

再由式 (19.6) 的 $y_k=a_ky_0+\sigma_k\varepsilon$ 换算出

$$
\widehat y_0^{(k)}
\mathrel{=}
\frac{y_k-\sigma_k\widehat\varepsilon_k}{a_k}.
\tag{19.9}
$$

把式 (19.9) 的 $\widehat y_0^{(k)}$ 代入式 (19.8) 的后验均值，再使用式 (19.3) 化简，得到

$$
\mu_\theta(y_k,k)
\mathrel{=}
\frac1{\sqrt{\alpha_k}}
\left(
y_k-
\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\widehat\varepsilon_k
\right).
\tag{19.10}
$$

由式 (19.7) 的高斯后验和式 (19.10) 的模型均值可知，标准 DDPM 更新为

$$
\boxed{
y_{k-1}
\mathrel{=}
\frac1{\sqrt{\alpha_k}}
\left(
y_k-
\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\widehat\varepsilon_k
\right)
\mathbin{+}
\sqrt{\widetilde\beta_k}\,z_k,
\qquad
z_k\sim\mathcal N(0,\mathrm{Id}).
}
\tag{19.11}
$$

这就是 DDPM 使用的有限步离散方式：它不是简单冻结连续 SDE 的漂移，而是保留已经定义好的离散前向高斯链的后验均值和后验方差。这里的“精确后验”只针对离散前向链且条件中包含真实 $y_0$；用网络估计替代 $y_0$ 后仍然存在模型误差。

### 19.2 DDPM 的一阶展开与反向 VP SDE 一致

在远离 $\tau=0$ 的内部时刻，当 $h_k\to0$ 时，

$$
\frac1{\sqrt{\alpha_k}}
\mathrel{=}
\frac1{\sqrt{1-\beta_k}}
\mathrel{=}
1+\frac12\beta_k+O(\beta_k^2),
$$

并且

$$
\widetilde\beta_k
\mathrel{=}
\beta_k+O(\beta_k^2).
$$

将式 (19.11) 展开到一阶，再使用式 (19.3) 中 $\sigma_k=\sqrt{1-\bar\alpha_k}$ 和式 (19.4) 中 $\beta_k=\beta(\tau_k)h_k+O(h_k^2)$，得到

$$
\boxed{
y_{k-1}^{\mathrm{DDPM}}
\mathrel{=}
\left(1+\frac12\beta(\tau_k)h_k\right)y_k
\mathbin{-}
\frac{\beta(\tau_k)h_k}{\sigma_k}\widehat\varepsilon_k
\mathbin{+}
\sqrt{\beta(\tau_k)h_k}\,z_k
+O_{\mathrm{ms}}(h_k^{3/2})+O(h_k^2)
}
\tag{19.12}
$$

其中 $O_{\mathrm{ms}}$ 表示随机增量的均方误差阶。将式 (17.7) 的反向 VP SDE 代入式 (18.5) 的 Euler--Maruyama 更新可知，式 (19.12) 正是其小步形式：

$$
\mathrm dy_\tau
\mathrel{=}
\left[
-\frac12\beta y_\tau
\mathbin{+}
\beta\frac{\varepsilon_\theta}{\sigma}
\right]\mathrm d\tau
\mathbin{+}
\sqrt\beta\,\mathrm d\overline W_\tau,
\qquad \tau:T\to0.
$$

因此应当区分：

- 标准 DDPM 的有限步公式来自离散高斯后验；
- 它在小步长下一阶收敛到反向 VP SDE。

### 19.3 DDIM 使用的离散方式：固定本步估计并按 schedule 重组

DDIM 可以复用 DDPM 训练得到的噪声预测器以及相同的固定时刻边际。由式 (19.6) 的信号—噪声分解可知，当前样本满足

$$
y_k
\mathrel{=}
a_k\widehat y_0^{(k)}
\mathbin{+}
\sigma_k\widehat\varepsilon_k,
$$

其中

$$
\widehat\varepsilon_k
\mathrel{=}
\varepsilon_\theta(y_k,k),
\qquad
\widehat y_0^{(k)}
\mathrel{=}
\frac{y_k-\sigma_k\widehat\varepsilon_k}{a_k}.
\tag{19.13}
$$

确定性 DDIM 在当前一步 $k\to k-1$ 内固定

$$
\widehat y_0^{(k)},
\qquad
\widehat\varepsilon_k,
$$

只把已知的 schedule 系数从 $(a_k,\sigma_k)$ 变到 $(a_{k-1},\sigma_{k-1})$。因此，将式 (19.13) 在当前步内固定，并把 schedule 系数换成下一时刻的系数，得到标准确定性 DDIM 更新

$$
\boxed{
y_{k-1}
\mathrel{=}
a_{k-1}\widehat y_0^{(k)}
\mathbin{+}
\sigma_{k-1}\widehat\varepsilon_k.
}
\tag{19.14}
$$

它没有在这一步重新采样新的随机变量。到下一步时，网络仍会在新的 $y_{k-1}$ 上重新计算 $\widehat\varepsilon_{k-1}$；所以“固定”只表示在当前一步内固定，并不是整条采样轨迹只调用一次网络。

这种跨步规则与 VP 概率流 ODE 的相容性可以通过代理曲线看出。在当前一步内构造

$$
y_\tau^{(k)}
\mathrel{=}
a(\tau)\widehat y_0^{(k)}
\mathbin{+}
\sigma(\tau)\widehat\varepsilon_k.
\tag{19.15}
$$

固定 $\widehat y_0^{(k)}$ 和 $\widehat\varepsilon_k$ 后，对式 (19.15) 求导得到

$$
\frac{\mathrm d y_\tau^{(k)}}{\mathrm d\tau}
\mathrel{=}
\dot a(\tau)\widehat y_0^{(k)}
\mathbin{+}
\dot\sigma(\tau)\widehat\varepsilon_k.
\tag{19.16}
$$

这里

$$
\dot a(\tau):=\frac{\mathrm da(\tau)}{\mathrm d\tau},
\qquad
\dot\sigma(\tau):=\frac{\mathrm d\sigma(\tau)}{\mathrm d\tau}.
$$

对 VP SDE，

$$
a(\tau)
\mathrel{=}
\exp\left[-\frac12\int_0^\tau\beta(s)\,\mathrm ds\right],
\qquad
\sigma^2(\tau)=1-a^2(\tau).
$$

因此由式 (17.1) 和式 (17.3) 可知，下面的关系来自 VP schedule，而不是另外任意定义的：

$$
\boxed{
\dot a=-\frac12\beta a,
\qquad
\dot\sigma=\frac{\beta a^2}{2\sigma},
\qquad
a^2+\sigma^2=1.
}
\tag{19.17}
$$

将式 (19.17) 代入式 (19.16)，再使用式 (19.15) 合并括号，得到

$$
\begin{aligned}
\frac{\mathrm d y_\tau^{(k)}}{\mathrm d\tau}
&=
-\frac12\beta
\left(
a\widehat y_0^{(k)}+\sigma\widehat\varepsilon_k
\right)
\mathbin{+}
\frac{\beta}{2\sigma}\widehat\varepsilon_k
\\
&=
-\frac12\beta y_\tau^{(k)}
\mathbin{+}
\frac{\beta}{2\sigma}\widehat\varepsilon_k.
\end{aligned}
\tag{19.18}
$$

对比式 (17.9) 可知，式 (19.18) 正是把当前噪声预测冻结后的 VP 概率流 ODE 漂移。沿式 (19.15) 的代理曲线从 $\tau_k$ 走到 $\tau_{k-1}$，便得到式 (19.14)。

> **备注：求导式中的第二项为加号与反向采样时噪声减少的关系**
>
> 正向时间中 $\dot\sigma>0$；但反向采样的时间增量为 $\tau_{k-1}-\tau_k=-h_k<0$，所以 $\dot\sigma(\tau_{k-1}-\tau_k)<0$。噪声系数的减少来自反向时间增量，而不是在求导公式中人为添加负号。

### 19.4 DDIM 的一阶展开与概率流 ODE 一致

由式 (18.2) 可知 $\tau_{k-1}=\tau_k-h_k$。从式 (19.17) 出发，对反向一步展开 schedule：

$$
a_{k-1}
\mathrel{=}
a_k+\frac12\beta(\tau_k)a_kh_k+O(h_k^2),
$$

$$
\sigma_{k-1}
\mathrel{=}
\sigma_k-
\frac{\beta(\tau_k)a_k^2}{2\sigma_k}h_k
+O(h_k^2).
\tag{19.19}
$$

将式 (19.19) 代入式 (19.14) 的标准 DDIM 更新，并使用式 (19.13) 和式 (19.17) 中的关系

$$
a_k\widehat y_0^{(k)}
\mathrel{=}
y_k-\sigma_k\widehat\varepsilon_k,
\qquad
a_k^2+\sigma_k^2=1,
$$

得到

$$
\boxed{
y_{k-1}^{\mathrm{DDIM}}
\mathrel{=}
\left(1+\frac12\beta(\tau_k)h_k\right)y_k
\mathbin{-}
\frac{\beta(\tau_k)h_k}{2\sigma_k}\widehat\varepsilon_k
+O(h_k^2).
}
\tag{19.20}
$$

将式 (17.9) 按式 (18.3) 反向 Euler 离散可知，式 (19.20) 正是 VP 概率流 ODE

$$
\frac{\mathrm dy_\tau}{\mathrm d\tau}
\mathrel{=}
-\frac12\beta y_\tau
\mathbin{+}
\frac12\beta\frac{\varepsilon_\theta}{\sigma}
$$

的反向 Euler 一阶形式。

不过，标准 DDIM 的有限步更新通常不等于直接在原坐标使用普通 Euler。两者在有限步内冻结的对象不同：

- 原坐标 Euler 冻结整个当前速度；
- DDIM 冻结 $\widehat y_0^{(k)}$ 和 $\widehat\varepsilon_k$，并精确使用 schedule 的两个端点值。

网络输出一般会沿轨迹变化，所以 DDIM 不是完整概率流 ODE 在任意大步长下的精确解；它只是对 schedule 的变化处理得更有结构。当网格变细时，两者具有相同的连续极限。

### 19.5 有限步公式与连续极限不能混为一层

汇总式 (19.12) 和式 (19.20) 的一阶结果：

$$
\boxed{
\begin{aligned}
\text{DDPM}
&\approx
\left(1+\frac12\beta h\right)y
\mathbin{-}
\frac{\beta h}{\sigma}\varepsilon_\theta
\mathbin{+}
\sqrt{\beta h}\,z,
\\
\text{DDIM}
&\approx
\left(1+\frac12\beta h\right)y
\mathbin{-}
\frac{\beta h}{2\sigma}\varepsilon_\theta.
\end{aligned}
}
\tag{19.21}
$$

由式 (19.21) 可知，在小步长连续极限中，它们的区别为

$$
\boxed{
\text{DDPM}\to\text{DDIM}
\quad\Longleftrightarrow\quad
\begin{cases}
\text{去掉每一步新注入的随机噪声},\\
\text{score/noise 漂移系数从 }1\text{ 变成 }1/2.
\end{cases}
}
\tag{19.22}
$$

但在有限步层面，不能把标准 DDIM 理解成“直接删除 DDPM 的 $z_k$”。DDPM 的系数来自离散高斯后验，DDIM 的系数来自本步信号/噪声估计和 schedule 端点重组；二者保留了不同的高阶项。

---

# 第五部分：实验

实验进一步研究：路径噪声 $\gamma$、生成扩散强度 $\epsilon$、ODE/SDE 选择以及网络参数化会怎样影响结果。

必须先说明：这些实验是论文特定数据、网络和求解器下的结果，不构成“SDE 永远优于 ODE”或“某个 $\epsilon$ 对所有任务最优”的证明。

## 20. 二维 checkerboard：$\gamma$ 与 $\epsilon$ 的作用不同

论文从

$$
\rho_0=\mathcal N(0,\mathrm{Id})
$$

出发，学习一个边界较尖锐的二维 checkerboard 目标分布。$v$ 和 $s$ 都使用四层、每层 512 个单元的 ReLU 网络。训练后，论文分别用 ODE（$\epsilon=0$）以及

$$
\epsilon\in\{0.5,1.0,2.5\}
$$

的 SDE 生成 300,000 个样本，并用核密度估计比较结果。

![图 4：不同 $\gamma$ 与 $\epsilon$ 下的二维生成密度](images/paper-figure-07-qualitative.png)

*说明：列方向改变插值的 $\gamma(t)$，行方向改变生成动力学的 $\epsilon$。$\epsilon=0$ 是概率流 ODE，$\epsilon>0$ 是同边际 SDE。来源：[论文 v4, Figure 7](https://arxiv.org/abs/2303.08797v4)。*

这张图验证了两个旋钮的分工：

- $\gamma$ 在训练时改变中间边际和需要学习的场；
- $\epsilon$ 在生成时改变 ODE/SDE 的轨迹随机性；
- 对精确场，固定 $\gamma$ 后边际理论上不依赖 $\epsilon$；图中差异来自学习误差和数值误差；
- 在该实验中，SDE 通常优于 ODE，但 $\gamma(t)=\sqrt{t(1-t)}$ 对应的 ODE 已明显更好，ODE 与 SDE 的差距也最小；
- $\epsilon$ 较大时需要更小的数值步长，不能只比较随机强度而忽略计算预算。


> **备注：这里得到的结论**
>
> 能得出的结论是：在有限网络误差下，适量 SDE 随机性可能比纯 ODE 更稳健，而且 $\gamma$ 的路径设计会影响差距。不能得出“$\epsilon$ 越大越好”，因为大扩散需要更多求解步，且过强扩散也可能降低精度。

---

## 21. Oxford Flowers：同一个初始噪声可以继续分叉

论文还在 $128\times128$ Oxford Flowers 数据集上训练 one-sided 线性和三角插值，使用与 DDPM 类似的 U-Net 参数化 $b$、$s$ 或 denoiser。下面各列从相同初始噪声出发：ODE 使用 $\epsilon=0$，SDE 使用逐渐增大的 $\epsilon$。

![图 5：Oxford Flowers 上的 ODE 与 SDE 生成](images/paper-figure-13-flowers.png)

*说明：ODE 用自适应 dopri5；SDE 在 $\epsilon=1,2,4$ 时分别使用 2,000、2,500、4,000 个 Heun 步。固定初始噪声后，ODE 输出唯一；SDE 因生成途中持续加入布朗增量而能得到不同花朵，且图中多样性随 $\epsilon$ 增加。来源：[论文 v4, Figure 13](https://arxiv.org/abs/2303.08797v4)。*

![图 6：生成样本与训练集近邻](images/paper-figure-14-nearest-neighbors.png)

*说明：顶部是一个生成样本，底部是训练集中按 $\ell_1$ 距离找到的五个近邻。近邻在视觉上不同，作为模型没有直接复制该样本的辅助检查。来源：[论文 v4, Figure 14](https://arxiv.org/abs/2303.08797v4)。*


---

## 22. 从实验得到的实际选择建议

| 实际目标 | 更合适的起点 | 原因 |
|---|---|---|
| 少步、快速、固定 latent 可复现 | ODE / DDIM 类求解 | 无新增随机性，容易使用高阶求解器 |
| 同一条件下产生多个合理结果 | SDE | 固定初始状态后仍可随机分叉 |
| 有限模型误差下追求稳健质量 | 调过 $\epsilon$ 的 SDE | 论文实验显示适量扩散可能减轻过度集中 |
| 同时要速度与稳健性 | 混合采样 | 高噪声阶段保留随机性，接近数据端逐渐令 $\epsilon\to0$ |

最稳妥的工程方式是让同一个模型支持 ODE 与 SDE 两种采样器，再在验证集上联合选择：

$$
\boxed{
\gamma(t)\text{（训练边际路径）},
\quad
\epsilon(t)\text{（生成随机性）},
\quad
\text{求解步数与求解器}.
}
$$

论文实验支持“合适的非零 $\epsilon$ 可能优于 $\epsilon=0$”，但同时表明扩散强度过大或步数不足都会损害结果。因此不存在脱离模型误差和计算预算的统一最优 ODE/SDE 选择。

---

# 第六部分：最终统一视角

## 23. 从双端点公式到常见方法

### 23.1 Flow Matching 分支

从式 (2.1)

$$
x_t=I(t,x_0,x_1)+\gamma(t)z
$$

在式 (2.1) 中令

$$
\gamma(t)=0,
$$

由式 (2.1) 和式 (4.1) 可知

$$
x_t=I(t,x_0,x_1),
\qquad
Y_t=\partial_tI.
$$

再选择式 (13.1) 使用的线性 $I$：

$$
I=(1-t)x_0+tx_1,
$$

对上式求导得到

$$
Y_t=x_1-x_0,
$$

也就是 linear Flow Matching / Rectified Flow 的基础训练形式。

### 23.2 Diffusion 分支

从高斯基分布出发，取式 (14.1) 的 one-sided 线性插值

$$
x_t=\alpha(t)z+\beta(t)x_1.
$$

按式 (15.1) 反转为数据到噪声时间：

$$
y_\tau=a(\tau)y_0+\sigma(\tau)\varepsilon.
$$

再选择式 (17.1) 的 VP schedule：

$$
a^2+\sigma^2=1,
$$

先得到 VP diffusion 使用的固定时刻高斯边际；再额外选择与这些边际一致的 VP Markov 动力学，才得到：

$$
\boxed{
\begin{array}{c}
\text{另行选择前向 VP SDE}
\longrightarrow
\text{DDPM 前向链}
\\[1mm]
\text{score/noise 网络}
\longrightarrow
\begin{cases}
\text{反向 SDE / DDPM},\\
\text{概率流 ODE / DDIM}.
\end{cases}
\end{array}}
$$

---

## 24. 方法对照表

| 方法 | 在统一框架中选择什么 | 学习什么 | 怎样生成 |
|---|---|---|---|
| Stochastic Interpolant | 一般 $I,\nu,\gamma$ | 总速度 $b$，可选 score/denoiser | ODE 或同边际 SDE |
| Endpoint Flow Matching | $\gamma=0$ | $\partial_tI$ 的条件平均 | ODE |
| Linear Flow Matching | $I=(1-t)x_0+tx_1$ | $x_1-x_0$ 的条件平均 | ODE |
| Rectified Flow | 线性 FM 加重新配对/reflow | 更接近当前生成耦合的线性速度 | ODE |
| VP diffusion | one-sided Gaussian 路径与 VP schedule | noise、score 或等价 velocity | 反向 SDE或概率流 ODE |
| DDPM | 离散 VP 前向链 | 常用 noise prediction | 随机反向高斯链 |
| DDIM | 与 DDPM 共享训练边际和预测器 | 通常不重新训练 | 确定性离散更新 |

---

## 参考文献

1. Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797), arXiv:2303.08797v4.
2. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), ICLR 2023.
3. Xingchao Liu, Chengyue Gong, Qiang Liu, [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003), ICLR 2023.
4. Jonathan Ho, Ajay Jain, Pieter Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020.
5. Jiaming Song, Chenlin Meng, Stefano Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), ICLR 2021.
6. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole, [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), ICLR 2021.
