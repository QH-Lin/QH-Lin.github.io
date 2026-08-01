---
title: "流匹配RL: 从统一diffusion与flow matching谈起"
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

## 从“先设计概率路径”到“再构造生成动力学”

本文希望回答一个核心问题：

> 给定易采样的噪声分布和目标数据分布，怎样先设计一条连接二者的概率路径，再用 ODE 或 SDE 在生成阶段实现这条路径？

全文最重要的逻辑链是

$$
\boxed{
\begin{array}{c}
\text{训练时构造随机插值 }x_t
\\[1mm]
\Downarrow
\\[1mm]
\text{得到目标边际路径 }\rho(t)
\text{ 和随机速度标签 }Y_t
\\[1mm]
\Downarrow
\\[1mm]
u(t,x)=\mathbb E[Y_t\mid x_t=x]
\\[1mm]
\Downarrow
\\[1mm]
\begin{cases}
\text{ODE 产生的密度 }q(t)=\rho(t),\\
\text{适当 SDE 产生的密度 }p(t)=\rho(t).
\end{cases}
\end{array}}
$$

这里的关键不是说 ODE、SDE 和随机插值具有相同的单条轨迹，而是说：

$$
\boxed{
\text{它们可以具有相同的单时刻边际分布。}
}
$$

---

## 1. 三个层次必须分开

为了避免混淆，先区分全文中的三类对象。

| 层次 | 随机变量或过程 | 密度 | 作用 |
|---|---|---|---|
| 训练时设计的随机插值 | $x_t$ | $\rho(t,x)$ | 定义希望模型复现的概率路径 |
| 生成 ODE | $X_t$ | $q(t,x)$ | 从噪声出发进行确定性生成 |
| 生成 SDE | $Z_t$ | $p(t,x)$ | 从噪声出发进行随机生成 |

另外还要区分两类速度：

| 符号 | 含义 |
|---|---|
| $Y_t=\dot x_t$ | 一条随机插值轨迹的随机速度 |
| $u(t,x)=\mathbb E[Y_t\mid x_t=x]$ | 给定时间和位置后的条件平均速度场 |

其中：

- $\rho,q,p$ 都是概率密度，不是速度场；
- $Y_t$ 是依赖隐藏变量的随机速度；
- $u(t,x)$ 是只依赖当前时间和位置的确定性速度场；
- 理论中的 $u$ 是精确条件期望，实际模型学习的是近似 $u_\theta$。

---

## 2. 第一步：用随机插值设计目标概率路径

### 2.1 两个端点与耦合

约定生成时间 $t$ 从 $0$ 增长到 $1$：

$$
\rho_0=\text{易采样的基分布，通常为高斯噪声},
\qquad
\rho_1=\text{目标数据分布}.
$$

采样

$$
(x_0,x_1)\sim\nu,
$$

其中 $\nu$ 的两个边际分别是 $\rho_0$ 和 $\rho_1$。最简单的选择是独立耦合

$$
\nu=\rho_0\otimes\rho_1.
$$

耦合 $\nu$ 决定哪些噪声样本与哪些数据样本配对，但不改变两个端点分布。

### 2.2 双端点随机插值

取满足端点条件的确定性插值函数

$$
I(0,x_0,x_1)=x_0,
\qquad
I(1,x_0,x_1)=x_1.
$$

再采样独立高斯潜变量

$$
z\sim\mathcal N(0,\mathrm{Id}),
\qquad
z\perp(x_0,x_1),
$$

定义

$$
\boxed{
x_t=I(t,x_0,x_1)+\gamma(t)z.
}
\tag{2.1}
$$

对本节的双端点随机插值，要求

$$
\gamma(0)=\gamma(1)=0.
$$

于是

$$
x_0\sim\rho_0,
\qquad
x_1\sim\rho_1.
$$

定义 $x_t$ 的边际密度为

$$
\boxed{
x_t\sim\rho(t,\cdot).
}
\tag{2.2}
$$

因此 $\rho(t)$ 是由随机插值预先设计出来的目标概率路径。此时它还没有被证明是某个生成 ODE 或生成 SDE 的密度。

### 2.3 为什么随机插值本身通常不是生成算法

训练时可以同时取得噪声样本 $x_0$ 和数据样本 $x_1$，所以容易构造 $x_t$。

但生成时只有

$$
x_0\sim\rho_0,
$$

并不知道目标端点 $x_1$。因此不能直接依靠式 (2.1) 生成新数据。

随机插值的主要作用是：

1. 定义希望生成模型复现的中间边际 $\rho(t)$；
2. 提供可以直接采样的监督信号；
3. 帮助构造只依赖 $(t,x)$ 的 ODE 或 SDE。

---

## 3. 从随机轨迹速度得到确定速度场

对式 (2.1) 关于时间求导：

$$
\boxed{
Y_t:=\dot x_t
\mathrel{=}
\partial_tI(t,x_0,x_1)+\dot\gamma(t)z.
}
\tag{3.1}
$$

$Y_t$ 是单条随机插值轨迹的速度。即使给定 $x_t=x$，仍可能有多组不同的 $(x_0,x_1,z)$ 经过该位置，并产生不同的 $Y_t$。

而 ODE 的速度场必须满足：给定 $(t,x)$ 后，速度是一个确定向量。

因此定义

$$
\boxed{
u(t,x)
:=
\mathbb E[Y_t\mid x_t=x].
}
\tag{3.2}
$$

这一步可以理解为从“拉格朗日随机轨迹速度”得到“欧拉确定速度场”。

需要特别注意：

$$
Y_t=u(t,x_t)
$$

一般不会逐样本成立。成立的是条件平均关系

$$
\boxed{
\mathbb E[Y_t\mid x_t]=u(t,x_t).
}
\tag{3.3}
$$

---

## 4. 为什么随机插值密度 $\rho$ 满足连续性方程

### 4.1 测试函数证明

为突出主要逻辑，以下假设 $I$ 和 $\gamma$ 对时间可微、相关期望有限、允许交换时间求导与期望，并假设涉及的密度和速度场足够光滑。

取任意光滑紧支撑测试函数

$$
\varphi\in C_c^\infty(\mathbb R^d).
$$

> **备注：什么是测试函数？**
>
> $\varphi\in C_c^\infty(\mathbb R^d)$ 表示 $\varphi$ 是定义在 $d$ 维空间上的光滑函数：$C^\infty$ 表示可以无限次求导，右下角的 $c$ 表示它只在某个有限区域内非零，称为“紧支撑”。
>
> 可以把 $\varphi$ 看成一个光滑探测器。若它在区域 $A$ 内接近 $1$、区域外接近 $0$，那么 $\mathbb E[\varphi(x_t)]$ 就近似测量 $x_t$ 落在区域 $A$ 内的概率。紧支撑还保证分部积分时无穷远处不会产生边界项。

由链式法则，

$$
\frac{\mathrm d}{\mathrm dt}\varphi(x_t)
\mathrel{=}
\nabla\varphi(x_t)\cdot Y_t.
$$

> **备注：为什么链式法则给出这个等式？**
>
> 写成坐标形式，$x_t=(x_t^{(1)},\ldots,x_t^{(d)})$。多变量链式法则给出：$\displaystyle \frac{\mathrm d}{\mathrm dt}\varphi(x_t)=\sum_{i=1}^d\frac{\partial\varphi}{\partial x_i}(x_t)\frac{\mathrm dx_t^{(i)}}{\mathrm dt}$。
>
> 梯度是 $\nabla\varphi=(\partial_{x_1}\varphi,\ldots,\partial_{x_d}\varphi)$，而 $Y_t=\dot x_t$，所以上面的求和正是点积 $\nabla\varphi(x_t)\cdot Y_t$。虽然 $x_t$ 是随机的，但固定一次随机采样后，它就是一条普通的时间轨迹，因此可以逐轨迹使用链式法则。

取期望：

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(x_t)]
\mathrel{=}
\mathbb E[
\nabla\varphi(x_t)\cdot Y_t
].
\tag{4.1}
$$

对 $x_t$ 使用条件期望：

$$
\begin{aligned}
\mathbb E[
\nabla\varphi(x_t)\cdot Y_t
]
&=
\mathbb E\!\left[
\mathbb E\!\left[
\nabla\varphi(x_t)\cdot Y_t
\mid x_t
\right]
\right]
\\
&=
\mathbb E\!\left[
\nabla\varphi(x_t)\cdot
\mathbb E[Y_t\mid x_t]
\right]
\\
&=
\mathbb E[
\nabla\varphi(x_t)\cdot u(t,x_t)
]
\\
&=
\int_{\mathbb R^d}
\nabla\varphi(x)\cdot u(t,x)\rho(t,x)\,\mathrm dx.
\end{aligned}
\tag{4.2}
$$

> **备注：第一个等号使用塔式法则，第二个等号使用条件期望的“已知量提出”性质。**
>
> 条件期望 $\mathbb E[Y_t\mid x_t]$ 可以理解为：按照 $x_t$ 的取值把样本分组，然后在每一组内对 $Y_t$ 求平均。
>
> **塔式法则：** 先在每个 $x_t$ 分组内求平均，再对所有分组求总体平均，等于直接求总体平均：$\mathbb E[Z]=\mathbb E\!\left[\mathbb E[Z\mid x_t]\right]$。
>
> **已知量提出：** 给定 $x_t$ 后，任何只依赖 $x_t$ 的量都已经确定。因此对任意函数 $g$，有 $\mathbb E[g(x_t)Y_t\mid x_t]=g(x_t)\mathbb E[Y_t\mid x_t]$。
>
> 这里 $g(x_t)=\nabla\varphi(x_t)$，所以 $\mathbb E[\nabla\varphi(x_t)\cdot Y_t]=\mathbb E[\nabla\varphi(x_t)\cdot\mathbb E[Y_t\mid x_t]]$。
>
> 最后使用 $u(t,x)=\mathbb E[Y_t\mid x_t=x]$，便得到式 (4.2)。

分部积分得到

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(x_t)]
\mathrel{=}
-\int_{\mathbb R^d}
\varphi(x)\nabla\cdot(\rho u)(t,x)\,\mathrm dx.
\tag{4.3}
$$

> **备注：分部积分 **
>
> 式 (4.2) 的最后一项是

$$
\int_{\mathbb R^d}\nabla\varphi(x)\cdot(\rho u)(t,x)\,\mathrm dx.
$$

> 分部积分的作用，是把原来作用在 $\varphi$ 上的空间导数 $\nabla$ 转移到 $\rho u$ 上；转移后会产生一个负号。
>
> 以一维情形为例。由乘积求导公式，

$$
(\varphi(x)F(x))'
\mathrel{=}
\varphi'(x)F(x)+\varphi(x)F'(x).
$$

> 在区间 $[a,b]$ 上对等式两边积分，得到

$$
\int_a^b(\varphi F)'(x)\,\mathrm dx
\mathrel{=}
\int_a^b\varphi'(x)F(x)\,\mathrm dx
\mathbin{+}
\int_a^b\varphi(x)F'(x)\,\mathrm dx.
$$

> 根据微积分基本定理，左边等于两个端点处函数值之差：

$$
\int_a^b(\varphi F)'(x)\,\mathrm dx
\mathrel{=}
\varphi(b)F(b)-\varphi(a)F(a).
$$

> 因而

$$
\int_a^b\varphi'(x)F(x)\,\mathrm dx
\mathrel{=}
\underbrace{\varphi(b)F(b)-\varphi(a)F(a)}_{\text{边界项}}
\mathbin{-}
\int_a^b\varphi(x)F'(x)\,\mathrm dx.
$$

> 这就是一维分部积分公式。负号来自把 $\int_a^b\varphi(x)F'(x)\,\mathrm dx$ 移到等号右边，并不是额外规定的。
>
> 把区间扩展到整个实数轴，就有

$$
\int_{\mathbb R}\varphi'(x)F(x)\,\mathrm dx
\mathrel{=}
[\varphi(x)F(x)]_{-\infty}^{+\infty}
\mathbin{-}
\int_{\mathbb R}\varphi(x)F'(x)\,\mathrm dx.
$$

> 如果 $\varphi$ 在足够远处为零，或者更一般地，$\varphi(x)F(x)$ 在 $x\to\pm\infty$ 时衰减到零，那么边界项消失，于是

$$
\int_{\mathbb R}\varphi'(x)F(x)\,\mathrm dx
\mathrel{=}
-\int_{\mathbb R}\varphi(x)F'(x)\,\mathrm dx.
$$

>
> 最后代入 $F=\rho u$，便由式 (4.2) 得到式 (4.3)。直观上，$\rho u$ 是概率流量，$\nabla\cdot(\rho u)>0$ 表示某处流出的概率多于流入的概率，因此该处的概率密度应当下降；这也解释了负号的物理意义。

另一方面，

$$
\mathbb E[\varphi(x_t)]
\mathrel{=}
\int_{\mathbb R^d}
\varphi(x)\rho(t,x)\,\mathrm dx,
$$

所以

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(x_t)]
\mathrel{=}
\int_{\mathbb R^d}
\varphi(x)\partial_t\rho(t,x)\,\mathrm dx.
\tag{4.4}
$$

比较式 (4.3) 和式 (4.4)：

$$
\int_{\mathbb R^d}
\varphi(x)
\left[
\partial_t\rho+\nabla\cdot(\rho u)
\right](t,x)\,\mathrm dx
=0.
$$

由于测试函数 $\varphi$ 可以任意选择，在上述光滑假设下，只能有

$$
\boxed{
\partial_t\rho(t,x)
\mathbin{+}
\nabla\cdot\bigl(\rho(t,x)u(t,x)\bigr)
=0
}
\tag{4.5}
$$

### 4.2 这一步表达了什么

随机插值的单条轨迹使用随机速度 $Y_t$，但其边际密度只感受到条件平均后的概率流量

$$
\boxed{
J_\rho(t,x)=\rho(t,x)u(t,x).
}
\tag{4.6}
$$

连续性方程就是概率质量守恒：

$$
\text{局部密度变化}
\mathbin{+}
\text{净流出量}
=0.
$$

---

## 5. 构造 ODE，并证明其密度 $q$ 等于 $\rho$

### 5.1 ODE 中的 $u$ 从哪里来

为了避免误解，暂时把式 (3.2) 得到的速度记作

$$
u_{\mathrm{SI}}(t,x)
\mathrel{=}
\mathbb E[Y_t\mid x_t=x].
$$

接下来人为构造 ODE：

$$
\boxed{
\dot X_t=u_{\mathrm{SI}}(t,X_t),
\qquad
X_0\sim\rho_0.
}
\tag{5.1}
$$

逻辑是

$$
\boxed{
\text{先从随机插值得到 }u_{\mathrm{SI}},
\quad
\text{再把同一个 }u_{\mathrm{SI}}\text{ 放入 ODE。}
}
$$

后文重新把 $u_{\mathrm{SI}}$ 简写为 $u$。

定义 ODE 解的密度：

$$
\boxed{
X_t\sim q(t,\cdot).
}
\tag{5.2}
$$

这里 $q$ 是 ODE 粒子群的密度。

### 5.2 为什么 $q$ 也满足连续性方程

因为

$$
\dot X_t=u(t,X_t),
$$

所以对测试函数 $\varphi$，

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(X_t)]
&=
\mathbb E\!\left[
\frac{\mathrm d}{\mathrm dt}\varphi(X_t)
\right]
\\
&=
\mathbb E[
\nabla\varphi(X_t)\cdot\dot X_t
]
\\
&=
\mathbb E[
\nabla\varphi(X_t)\cdot u(t,X_t)
]
\\
&=
\int_{\mathbb R^d}
\nabla\varphi(x)\cdot u(t,x)q(t,x)\,\mathrm dx
\\
&=
-\int_{\mathbb R^d}
\varphi(x)\nabla\cdot(qu)(t,x)\,\mathrm dx.
\end{aligned}
\tag{5.3}
$$

另一方面，

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(X_t)]
\mathrel{=}
\int_{\mathbb R^d}
\varphi(x)\partial_tq(t,x)\,\mathrm dx.
\tag{5.4}
$$

因此

$$
\boxed{
\partial_tq+\nabla\cdot(qu)=0.
}
\tag{5.5}
$$

这个证明与第 4 节的数学骨架相同。区别是：

- 对随机插值，$\dot x_t=Y_t$ 不是 $x_t$ 的确定函数，需要条件平均；
- 对 ODE，$\dot X_t=u(t,X_t)$ 已经是当前位置的确定函数。

#### 两个证明的核心差异

> 下式中，黑色部分是两个证明共有的操作；红色部分才是两者真正不同的地方。

$$
\boxed{
\begin{aligned}
\text{随机插值：}\quad
\frac{\mathrm d}{\mathrm dt}\mathbb E[\varphi(x_t)]
&=
\mathbb E[
\nabla\varphi(x_t)\cdot
Y_t
]
\\
&\mathrel{\overset{
{\color{red}{\text{对 }x_t\text{ 做条件平均}}}
}{=}}
\mathbb E[
\nabla\varphi(x_t)\cdot
\mathbb E[Y_t\mid x_t]
]
\\
&=
\mathbb E[
\nabla\varphi(x_t)\cdot u(t,x_t)
],
\\[3mm]
\text{ODE：}\quad
\frac{\mathrm d}{\mathrm dt}\mathbb E[\varphi(X_t)]
&=
\mathbb E[
\nabla\varphi(X_t)\cdot
\dot X_t
]
\\
&\mathrel{\overset{
{\color{red}{\dot X_t=u(t,X_t)}}
}{=}}
\mathbb E[
\nabla\varphi(X_t)\cdot u(t,X_t)
].
\end{aligned}
}
\tag{5.5a}
$$

**两种方法的物理意义：**

- **随机插值描述微观的随机轨迹。** 即使多条轨迹在同一时刻到达同一位置 $x_t=x$，它们的瞬时速度 $Y_t$ 也可能不同，因为它们可能来自不同的端点或噪声样本。边际密度并不分别追踪这些不同速度，只感受到它们在当前位置的条件平均
  $$
  u(t,x)=\mathbb E[Y_t\mid x_t=x].
  $$
  因而，$u$ 表示位置 $x$ 处概率质量的平均输运方向；速度偏差 $Y_t-u(t,x_t)$ 在同一位置进行条件平均后相互抵消。

- **ODE 描述宏观的确定性概率流。** ODE 直接规定每个位置上的粒子速度为 $u(t,x)$。所有到达同一位置的 ODE 粒子都使用同一个速度场，不再保留随机插值中“同一位置可能具有不同瞬时速度”的微观差异。因此，ODE 可以看成直接实现条件平均后的概率流。

两者的微观轨迹可以不同，但只要使用同一个速度场 $u$，它们对边际密度产生的概率流分别是 $\rho u$ 和 $qu$，并满足相同形式的连续性方程。随机插值负责从随机轨迹中提取平均速度场，ODE 则使用这个平均速度场输运概率质量。

红色文字就是唯一的关键区别：随机插值需要先对随机速度做条件平均；ODE 直接使用定义中的确定速度。完成这一步以后，两边都变成“密度由同一个速度场 $u$ 输运”，后续的分部积分完全相同：

$$
\boxed{
\begin{aligned}
\text{随机插值：}\quad&
\partial_t\rho
\mathbin{+}
\nabla\cdot(\rho u)=0,
\\
\text{ODE：}\quad&
\partial_tq
\mathbin{+}
\nabla\cdot(qu)=0.
\end{aligned}
}
\tag{5.5b}
$$

事实上

$$
\mathbb E[\dot X_t\mid X_t=x]
\mathrel{=}
\mathbb E[u(t,X_t)\mid X_t=x]
\mathrel{=}
u(t,x).
$$

所以 ODE 是一般“条件平均速度定理”的特殊情况。

### 5.3 为什么能推出 $q=\rho$

目前已经分别证明

$$
\partial_t\rho+\nabla\cdot(\rho u)=0,
\qquad
\rho(0)=\rho_0,
\tag{5.6}
$$

以及

$$
\partial_tq+\nabla\cdot(qu)=0,
\qquad
q(0)=\rho_0.
\tag{5.7}
$$

在 $u$ 对空间变量具有足够正则性，例如 Lipschitz，并满足适当增长条件时，连续性方程的初值问题具有唯一解。因此

$$
\boxed{
q(t,x)=\rho(t,x).
}
\tag{5.8}
$$

特别地，

$$
X_1\sim q(1,\cdot)=\rho(1,\cdot)=\rho_1.
$$

这正是生成建模需要的结论：

> 训练时借助已知数据端点设计随机插值；生成时不再使用数据端点，只从噪声出发运行 ODE，仍能在理想条件下到达目标数据分布。

需要注意，$q=\rho$ 只表示单时刻边际相同。随机插值轨迹 $x_t$ 与 ODE 轨迹 $X_t$ 通常不同。

---

## 6. 同一条 $\rho(t)$ 也可以由 SDE 实现

### 6.1 本章要证明什么

第 4 章已经从随机插值得到目标边际密度 $\rho(t,x)$ 和条件平均速度场 $u(t,x)$，并证明它们满足连续性方程

$$
\boxed{
\partial_t\rho(t,x)
\mathbin{+}
\nabla\cdot\bigl(\rho(t,x)u(t,x)\bigr)
\mathrel{=}
0,
\qquad
\rho(0,x)=\rho_0(x).
}
\tag{6.1}
$$

第 5 章进一步说明：由同一个速度场驱动的 ODE

$$
\dot X_t=u(t,X_t)
$$

可以实现这条边际密度路径。

本章要回答一个新的问题：

> 能否在轨迹中加入布朗噪声，同时仍让每个时刻的边际密度保持为 $\rho(t,x)$？

答案是可以，但不能只在 ODE 上直接加噪声。随机扩散会摊平密度，因此还需要在漂移中加入一个由 score 决定的补偿项。

本章将构造 SDE $Z_t$，记它的实际密度为 $p(t,x)$，并证明

$$
\boxed{
p(t,x)=\rho(t,x).
}
$$

全文需要区分三个随机对象：

| 随机对象 | 如何产生 | 时刻 $t$ 的密度 |
|---|---|---|
| $x_t$ | 训练时设计的随机插值 | $\rho(t,x)$ |
| $X_t$ | 速度场 $u$ 驱动的 ODE | $q(t,x)$ |
| $Z_t$ | 本章构造的 SDE | $p(t,x)$ |

证明路线是：

1. 先推导一般 SDE 的 Fokker--Planck 方程；
2. 再要求它在密度为 $\rho$ 时具有与 ODE 相同的概率流量 $\rho u$；
3. 由此反推出合适的漂移；
4. 最后用相同 PDE、相同初值和解的唯一性证明 $p=\rho$。

### 6.2 Itô SDE 的符号与尺度

#### 随机状态、空间位置和密度

- $Z_t\in\mathbb R^d$ 是时刻 $t$ 的随机状态；
- $x\in\mathbb R^d$ 是确定的空间位置，是密度和向量场的自变量；
- $p(t,x)$ 是 $Z_t$ 的实际密度，即 $Z_t\sim p(t,\cdot)$；
- $\rho(t,x)$ 是随机插值预先规定的目标密度。

在证明结束以前，$p$ 和 $\rho$ 必须严格区分：$p$ 是 SDE 实际产生的未知密度，$\rho$ 是希望它实现的候选密度。

#### 布朗运动

令 $W_t\in\mathbb R^d$ 为 $d$ 维标准 Wiener 过程，也称标准布朗运动。对有限的小时间步 $\Delta t>0$，

$$
\Delta W_t
:=
W_{t+\Delta t}-W_t
\sim
\mathcal N(0,\Delta t\,\mathrm{Id}),
$$

其中 $\mathrm{Id}$ 是 $d\times d$ 单位矩阵。等价地，

$$
\Delta W_t
\overset{\mathrm d}{=}
\sqrt{\Delta t}\,\xi,
\qquad
\xi\sim\mathcal N(0,\mathrm{Id}),
$$

这里 $\overset{\mathrm d}{=}$ 表示两边分布相同。

> **备注：为什么这两种写法等价？**
>
> 第一种写法来自标准布朗运动的定义：长度为 $\Delta t$ 的时间区间对应一个均值为零、协方差为 $\Delta t\,\mathrm{Id}$ 的高斯增量。
>
> 令 $\xi\sim\mathcal N(0,\mathrm{Id})$，并定义

$$
Y:=\sqrt{\Delta t}\,\xi.
$$

> 高斯向量乘以常数后仍是高斯向量。它的均值为

$$
\mathbb E[Y]
\mathrel{=}
\sqrt{\Delta t}\,\mathbb E[\xi]
=0,
$$

> 协方差为

$$
\begin{aligned}
\operatorname{Cov}(Y)
&=
\mathbb E[YY^{\mathsf T}]
\\
&=
\Delta t\,\mathbb E[\xi\xi^{\mathsf T}]
\\
&=
\Delta t\,\mathrm{Id}.
\end{aligned}
$$

> 因此

$$
\sqrt{\Delta t}\,\xi
\sim
\mathcal N(0,\Delta t\,\mathrm{Id}),
$$

> 与 $\Delta W_t$ 具有完全相同的分布，所以可以写成

$$
\Delta W_t
\overset{\mathrm d}{=}
\sqrt{\Delta t}\,\xi.
$$

> 从坐标上看也一样：$\xi$ 的每个分量都是相互独立的 $\mathcal N(0,1)$；乘以 $\sqrt{\Delta t}$ 后，每个分量都变成 $\mathcal N(0,\Delta t)$。
>
> 这也是数值模拟布朗运动的方法：每走一个长度为 $\Delta t$ 的时间步，就重新采样一个独立的标准高斯向量 $\xi$，再乘以 $\sqrt{\Delta t}$ 作为该步的布朗增量。
>
> 需要注意，$\overset{\mathrm d}{=}$ 只表示两边的概率分布相同，不表示两个随机变量在每一次采样时都取到相同数值。它说明的是：可以用一个标准高斯向量 $\xi$ 生成一个与布朗增量同分布的样本。

因此布朗增量是 $\sqrt{\Delta t}$ 量级，而不是 $\Delta t$ 量级。符号 $\mathrm dW_t$ 是布朗增量的连续时间记号，不能理解成 $\dot W_t\,\mathrm dt$，因为布朗轨迹几乎处处不可微。

#### 漂移和噪声幅度

本章采用 Itô 解释，先把 SDE 写成最直观的形式：

$$
\boxed{
\mathrm dZ_t
\mathrel{=}
b(t,Z_t)\,\mathrm dt
\mathbin{+}
\sigma(t)\,\mathrm dW_t,
\qquad
\sigma(t)\ge0.
}
\tag{6.2}
$$

除非特别说明，本章后面出现的所有 SDE 和随机积分都按 Itô 意义理解。

式中：

- $b(t,x)\in\mathbb R^d$ 是漂移速度，控制随机状态的平均移动方向；
- $\sigma(t)\ge0$ 是直接乘在布朗增量上的**噪声幅度**；
- $\sigma(t)=0$ 表示该时刻不加入新噪声；
- $\sigma(t)$ 越大，单位时间内随机增量的方差越大，单条轨迹越随机。

为了简化推导，本章只考虑 $\sigma$ 依赖时间、且所有空间方向使用相同噪声幅度的情形，因此扩散是各向同性的。

> **备注：什么是 Itô 型 SDE？**
>
> “Itô 型 SDE”指其中的随机积分按照 Itô 规则定义的随机微分方程。更一般的形式为

$$
\mathrm dZ_t
\mathrel{=}
b(t,Z_t)\,\mathrm dt
\mathbin{+}
\sigma(t,Z_t)\,\mathrm dW_t.
$$

> 它严格表示积分方程

$$
Z_t
\mathrel{=}
Z_0
\mathbin{+}
\int_0^t b(r,Z_r)\,\mathrm dr
\mathbin{+}
\int_0^t\sigma(r,Z_r)\,\mathrm dW_r.
$$

> 第一个积分是普通时间积分，最后一项按照 Itô 随机积分解释。
>
> Itô 规则的核心是使用每个小区间的左端点。把 $[0,t]$ 分成小区间 $[t_k,t_{k+1}]$，则随机积分由下列和式的极限定义：

$$
\int_0^t\sigma(r,Z_r)\,\mathrm dW_r
\approx
\sum_k
\sigma(t_k,Z_{t_k})
\bigl(W_{t_{k+1}}-W_{t_k}\bigr).
$$

> 因此，在计算下一步的随机增量时，系数使用当前已经知道的状态 $Z_{t_k}$：

$$
Z_{t_{k+1}}-Z_{t_k}
\approx
b(t_k,Z_{t_k})\Delta t
\mathbin{+}
\sigma(t_k,Z_{t_k})\sqrt{\Delta t}\,\xi_k,
\qquad
\xi_k\sim\mathcal N(0,\mathrm{Id}).
$$

> 当前的系数不依赖尚未发生的未来布朗增量。正因为采用左端点，满足适当可积性条件时，Itô 随机积分具有零均值性质：

$$
\mathbb E\!\left[
\int_0^t\sigma(r,Z_r)\,\mathrm dW_r
\right]
=0.
$$

在一个很小但有限的时间步内，给定 $Z_t=x$，Euler--Maruyama 近似进行数值求解

$$
Z_{t+\Delta t}-Z_t
\approx
b(t,x)\Delta t
\mathbin{+}
\sigma(t)\sqrt{\Delta t}\,\xi,
\qquad
\xi\sim\mathcal N(0,\mathrm{Id}).
\tag{6.3}
$$

所以

$$
\begin{aligned}
\mathbb E[Z_{t+\Delta t}-Z_t\mid Z_t=x]
&\approx
b(t,x)\Delta t,
\\
\operatorname{Cov}(Z_{t+\Delta t}-Z_t\mid Z_t=x)
&\approx
\sigma^2(t)\Delta t\,\mathrm{Id}.
\end{aligned}
\tag{6.4}
$$

这说明 $b$ 控制平均位移，而 $\sigma$ 直接控制随机增量的标准差。

### 6.3 先得到一般 SDE 的 Fokker--Planck 方程

先给出本节结论。对于式 (6.2)

$$
\mathrm dZ_t
\mathrel{=}
b(t,Z_t)\,\mathrm dt
\mathbin{+}
\sigma(t)\,\mathrm dW_t,
$$

如果 $Z_t$ 的密度为 $p(t,x)$，那么它满足

$$
\boxed{
\partial_t p
\mathrel{=}
-\nabla\cdot(bp)
\mathbin{+}
\frac12\sigma^2(t)\Delta p.
}
$$

这个方程称为 Fokker--Planck 方程。它描述的不是单条随机轨迹，而是大量 SDE 轨迹形成的概率密度如何随时间变化：

- $-\nabla\cdot(bp)$ 是漂移造成的概率输运；
- $\tfrac12\sigma^2\Delta p$ 是噪声造成的概率扩散。

下面证明这个结论。证明链只有四步：

$$
\text{Itô 公式}
\longrightarrow
\text{取期望}
\longrightarrow
\text{按密度积分并分部积分}
\longrightarrow
\text{得到密度方程}.
$$

取光滑、紧支撑的测试函数 $\varphi\in C_c^\infty(\mathbb R^d)$。Itô 公式给出

$$
\begin{aligned}
\mathrm d\varphi(Z_t)
={}&
\nabla\varphi(Z_t)\cdot b(t,Z_t)\,\mathrm dt
\\
&+
\frac12\sigma^2(t)\Delta\varphi(Z_t)\,\mathrm dt
\\
&+
\sigma(t)\nabla\varphi(Z_t)\cdot\mathrm dW_t.
\end{aligned}
\tag{6.5}
$$

将式 (6.5) 从 $0$ 积分到 $t$，再取期望。满足适当可积性条件时，Itô 随机积分的期望为零，因此

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(Z_t)]
={}&
\mathbb E[
\nabla\varphi(Z_t)\cdot b(t,Z_t)
]
\\
&+
\frac12\sigma^2(t)
\mathbb E[\Delta\varphi(Z_t)].
\end{aligned}
\tag{6.6}
$$

因为 $Z_t$ 的密度是 $p(t,x)$，可以把期望写成积分：

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(Z_t)]
={}&
\int_{\mathbb R^d}
\nabla\varphi(x)\cdot b(t,x)p(t,x)\,\mathrm dx
\\
&+
\frac12\sigma^2(t)
\int_{\mathbb R^d}
\Delta\varphi(x)p(t,x)\,\mathrm dx.
\end{aligned}
\tag{6.7}
$$

对第一项分部积分一次，对第二项分部积分两次。因为 $\varphi$ 具有紧支撑，边界项为零：

$$
\begin{aligned}
\int_{\mathbb R^d}\nabla\varphi\cdot(bp)\,\mathrm dx
&=
-\int_{\mathbb R^d}\varphi\,\nabla\cdot(bp)\,\mathrm dx,
\\
\int_{\mathbb R^d}(\Delta\varphi)p\,\mathrm dx
&=
\int_{\mathbb R^d}\varphi\,\Delta p\,\mathrm dx.
\end{aligned}
$$

于是

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(Z_t)]
\mathrel{=}
\int_{\mathbb R^d}
\varphi(x)
\left[
-\nabla\cdot(bp)
\mathbin{+}
\frac12\sigma^2\Delta p
\right](t,x)\,\mathrm dx.
\tag{6.8}
$$

另一方面，

$$
\mathbb E[\varphi(Z_t)]
\mathrel{=}
\int_{\mathbb R^d}\varphi(x)p(t,x)\,\mathrm dx,
$$

所以

$$
\frac{\mathrm d}{\mathrm dt}
\mathbb E[\varphi(Z_t)]
\mathrel{=}
\int_{\mathbb R^d}
\varphi(x)\partial_t p(t,x)\,\mathrm dx.
\tag{6.9}
$$

比较式 (6.8) 和式 (6.9)。由于测试函数 $\varphi$ 可以任意选择，只能有

$$
\boxed{
\partial_t p
\mathrel{=}
-\nabla\cdot(bp)
\mathbin{+}
\frac12\sigma^2\Delta p.
}
\tag{6.10}
$$

这就证明了本节开头给出的 Fokker--Planck 方程。

> **补充：式 (6.5) 的二阶项从哪里来？**
>
> Itô 公式可以形式化地看成二阶 Taylor 展开：

$$
\mathrm d\varphi(Z_t)
\mathrel{=}
\nabla\varphi(Z_t)\cdot\mathrm dZ_t
\mathbin{+}
\frac12
(\mathrm dZ_t)^{\mathsf T}
\nabla^2\varphi(Z_t)
\mathrm dZ_t.
$$

> 代入 $\mathrm dZ_t=b\,\mathrm dt+\sigma\,\mathrm dW_t$，并使用

$$
(\mathrm dt)^2=0,
\qquad
\mathrm dt\,\mathrm dW_t=0,
\qquad
\mathrm dW_t\mathrm dW_t^{\mathsf T}
\mathrel{=}
\mathrm{Id}\,\mathrm dt,
$$

> 二阶项只剩

$$
\frac12\sigma^2(t)
\operatorname{tr}(\nabla^2\varphi(Z_t))\,\mathrm dt
\mathrel{=}
\frac12\sigma^2(t)
\Delta\varphi(Z_t)\,\mathrm dt.
$$

> 因此式 (6.5) 比普通链式法则多出一个正的二阶项。

### 6.4 定义 score，并用它设计漂移

第 6.3 节表明，布朗噪声会在密度方程中产生扩散项。为了设计抵消这个扩散项的漂移，需要知道目标密度 $\rho(t,x)$ 在每个位置朝哪个方向增大。这个方向由 score 给出。

假设 $\rho(t,x)>0$ 且足够光滑，定义

$$
\boxed{
s(t,x)
:=
\nabla_x\log\rho(t,x).
}
\tag{6.11}
$$

score 是一个 $d$ 维向量。它的物理含义是：

- 方向：指向当前位置附近密度上升最快的方向；
- 大小：表示附近密度变化得有多快；
- 在密度的局部峰顶，score 为零。

由链式法则，

$$
s(t,x)
\mathrel{=}
\frac{\nabla_x\rho(t,x)}{\rho(t,x)}.
$$

因此

$$
\boxed{
\rho(t,x)s(t,x)
\mathrel{=}
\nabla_x\rho(t,x).
}
\tag{6.12}
$$

这个等式是后面证明漂移与扩散相互抵消的关键。

例如，若

$$
\rho(x)=\mathcal N(\mu,\alpha^2\mathrm{Id}),
$$

那么

$$
s(x)
\mathrel{=}
-\frac{x-\mu}{\alpha^2}
\mathrel{=}
\frac{\mu-x}{\alpha^2}.
$$

所以 Gaussian 的 score 总是指向分布中心 $\mu$：离中心越远，指向中心的作用越强。

需要注意两点：

1. 这里的 $s(t,x)$ 是目标密度 $\rho(t,x)$ 的 score，不是 SDE 实际密度 $p(t,x)$ 的 score。
2. score 本身不是速度。后面乘上由噪声幅度决定的系数 $\tfrac12\sigma^2(t)$ 后，
   $$
   \frac12\sigma^2(t)s(t,x)
   $$
   才成为漂移速度的一部分，用来抵消布朗扩散。

#### 从目标概率流反推出漂移

现在已经知道噪声会在密度方程中增加

$$
\frac12\sigma^2(t)\Delta p.
$$

这说明不能简单地在 ODE 上加噪声。若直接写成

$$
\mathrm dZ_t
\mathrel{=}
u(t,Z_t)\,\mathrm dt
\mathbin{+}
\sigma(t)\,\mathrm dW_t,
$$

那么其密度满足

$$
\partial_t p
\mathrel{=}
-\nabla\cdot(up)
\mathbin{+}
\frac12\sigma^2\Delta p,
$$

比目标连续性方程多出扩散项，通常会改变原来的密度路径。

从这里开始，为了简化公式，定义扩散率

$$
\boxed{
\kappa(t)
:=
\frac{\sigma^2(t)}{2}
\ge0,
\qquad
\sigma(t)=\sqrt{2\kappa(t)}.
}
$$

于是 Fokker--Planck 方程写成

$$
\partial_t p
\mathrel{=}
-\nabla\cdot(bp)
\mathbin{+}
\kappa\Delta p.
$$

因为 $\kappa$ 只依赖时间，

$$
\kappa\Delta p
\mathrel{=}
\nabla\cdot(\kappa\nabla p).
$$

所以方程可以改写为概率质量守恒形式

$$
\partial_t p+\nabla\cdot J_p=0,
\qquad
J_p
\mathrel{=}
bp-\kappa\nabla p.
$$

这里：

- $bp$ 是漂移流量；
- $-\kappa\nabla p$ 是扩散流量。

另一方面，目标密度 $\rho$ 已满足

$$
\partial_t\rho+\nabla\cdot(\rho u)=0,
$$

因此目标概率流量是 $\rho u$。

若希望 SDE 在密度为 $\rho$ 时具有同样的总流量，一个直接的选择是令

$$
\boxed{
b(t,x)\rho(t,x)
\mathbin{-}
\kappa(t)\nabla\rho(t,x)
\mathrel{=}
u(t,x)\rho(t,x).
}
\tag{6.13}
$$

利用 score 恒等式

$$
\frac{\nabla\rho}{\rho}
\mathrel{=}
\nabla\log\rho
\mathrel{=}
s,
$$

在 $\rho>0$ 的位置上将式 (6.13) 除以 $\rho$，得到

$$
\boxed{
b(t,x)
\mathrel{=}
u(t,x)+\kappa(t)s(t,x).
}
\tag{6.14}
$$

这一步的物理意义很简单：

$$
\underbrace{\kappa\rho s}_{+\kappa\nabla\rho\text{，score 漂移流量}}
\mathbin{+}
\underbrace{(-\kappa\nabla\rho)}_{\text{扩散流量}}
\mathrel{=}
0.
$$

因此，score 漂移抵消扩散流量，最后只剩目标流量 $\rho u$。

严格地说，只要额外概率流的散度为零就不会改变密度，所以漂移不绝对唯一。式 (6.14) 是最直接的逐点抵消选择。

### 6.5 构造 SDE，并证明它的密度等于 $\rho$

将式 (6.14) 和 $\sigma=\sqrt{2\kappa}$ 代回原 SDE，得到

$$
\boxed{
\mathrm dZ_t
\mathrel{=}
\bigl(u+\kappa s\bigr)(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\kappa(t)}\,\mathrm dW_t,
\qquad
Z_0\sim\rho_0.
}
\tag{6.15}
$$

记该 SDE 的实际密度为 $p(t,x)$。下面分三步证明 $p=\rho$。

**第一步：写出实际密度 $p$ 的方程。**

由 Fokker--Planck 方程，

$$
\begin{cases}
\partial_t p
\mathrel{=}
-\nabla\cdot\bigl((u+\kappa s)p\bigr)
\mathbin{+}
\kappa\Delta p,
\\
p(0,x)=\rho_0(x).
\end{cases}
$$

**第二步：验证目标密度 $\rho$ 也满足同一个方程。**

由式 (6.12)，

$$
\nabla\cdot(\rho s)
\mathrel{=}
\nabla\cdot(\nabla\rho)
\mathrel{=}
\Delta\rho.
\tag{6.16}
$$

因此

$$
\begin{aligned}
&-\nabla\cdot\bigl((u+\kappa s)\rho\bigr)
+\kappa\Delta\rho
\\
&\quad=
-\nabla\cdot(u\rho)
-\kappa\nabla\cdot(\rho s)
+\kappa\Delta\rho
\\
&\quad=
-\nabla\cdot(u\rho)
-\kappa\Delta\rho
+\kappa\Delta\rho
\\
&\quad=
-\nabla\cdot(u\rho)
\\
&\quad=
\partial_t\rho.
\end{aligned}
\tag{6.17}
$$

最后一个等号使用了目标连续性方程

$$
\partial_t\rho=-\nabla\cdot(\rho u).
$$

这里没有预先假设 $p=\rho$。我们只是把已知的候选函数 $\rho$ 代入 $p$ 所满足的 PDE，检查它确实也是一个解。

**第三步：使用相同初值和唯一性。**

现在 $p$ 和 $\rho$：

1. 满足同一个 Fokker--Planck 方程；
2. 具有同一个初值 $\rho_0$。

在 $u,s,\kappa$ 具有适当正则性、相应初值问题的解唯一时，

$$
\boxed{
p(t,x)=\rho(t,x).
}
\tag{6.18}
$$

特别地，

$$
Z_1\sim p(1,\cdot)=\rho(1,\cdot)=\rho_1.
$$

因此，式 (6.15) 的单条轨迹虽然带有随机性，但它在每个时刻的理想边际密度仍然是预先设计的 $\rho(t)$。

### 6.6 用概率流理解抵消机制

Fokker--Planck 方程可以写成概率质量守恒形式

$$
\partial_t p+\nabla\cdot J_p=0,
$$

其中

$$
\boxed{
J_p
\mathrel{=}
\underbrace{bp}_{\text{漂移流量}}
\mathbin{-}
\underbrace{\kappa\nabla p}_{\text{扩散流量的反向梯度形式}}.
}
\tag{6.19}
$$

更准确地说，扩散流量本身是 $-\kappa\nabla p$：它从高密度处流向低密度处。

当 $p=\rho$ 且 $b=u+\kappa s$ 时，

$$
\begin{aligned}
J_\rho
&=
(u+\kappa s)\rho-\kappa\nabla\rho
\\
&=
u\rho
\mathbin{+}
\underbrace{\kappa\rho s}_{\kappa\nabla\rho}
\mathbin{-}
\kappa\nabla\rho
\\
&=
u\rho.
\end{aligned}
\tag{6.20}
$$

因此三种作用可以这样理解：

| 作用 | 概率流量 | 物理效果 |
|---|---|---|
| 原始速度场 | $\rho u$ | 沿目标路径输运概率质量 |
| score 补偿漂移 | $+\kappa\nabla\rho$ | 从低密度方向推回高密度方向 |
| 布朗扩散 | $-\kappa\nabla\rho$ | 从高密度区域摊向低密度区域 |

后两项逐点抵消，所以 SDE 的总概率流仍然是 $\rho u$。这就是 ODE 和 SDE 可以拥有相同单时刻边际密度的最直观原因。

$\kappa$ 可以看成轨迹随机性的旋钮：

- $\kappa(t)=0$ 时，式 (6.15) 退化为概率流 ODE；
- $\kappa(t)>0$ 时，单条轨迹具有随机性，但理想边际仍然是 $\rho(t)$；
- 不同的非负函数 $\kappa(t)$ 给出不同的随机轨迹族，却可以共享同一条边际密度路径。

### 6.7 前向 SDE、反向 SDE 与 ODE

沿 $t:0\to1$ 运行的前向 SDE 是

$$
\mathrm dZ_t
\mathrel{=}
\bigl(u+\kappa s\bigr)(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\kappa(t)}\,\mathrm dW_t.
$$

它在时刻 $t$ 的 score 正是 $s(t,x)=\nabla\log\rho(t,x)$。对扩散系数 $\sqrt{2\kappa}$，反向时间漂移等于

$$
\begin{aligned}
b_{\mathrm{rev}}
&=
b_{\mathrm{fwd}}-2\kappa s
\\
&=
(u+\kappa s)-2\kappa s
\\
&=
u-\kappa s.
\end{aligned}
$$

因此，若仍使用变量 $t$，但从 $t=1$ 向 $t=0$ 积分，反向 SDE 写成

$$
\boxed{
\mathrm dZ_t
\mathrel{=}
\bigl(u-\kappa s\bigr)(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\kappa(t)}\,\mathrm d\overline W_t,
\qquad
t:1\longrightarrow0.
}
\tag{6.21}
$$

这里 $\overline W_t$ 表示反向时间 Wiener 过程。式 (6.21) 中的时间步 $\mathrm dt$ 为负；不能把它当作沿正时间方向运行的普通 SDE。

如果希望始终使用递增时间，可以定义反向时钟

$$
\tau=1-t,
\qquad
\widetilde Z_\tau=Z_{1-\tau}.
$$

那么 $\tau:0\to1$，相应方程为

$$
\mathrm d\widetilde Z_\tau
\mathrel{=}
\left[
-u+\kappa s
\right](1-\tau,\widetilde Z_\tau)\,\mathrm d\tau
\mathbin{+}
\sqrt{2\kappa(1-\tau)}\,\mathrm d\widetilde W_\tau.
$$

总结如下：

| 动力学 | 运行方向 | 漂移 | 单条轨迹 | 理想边际 |
|---|---|---|---|---|
| 概率流 ODE | $0\to1$ | $u$ | 给定初值后确定 | $\rho(t)$ |
| 前向 SDE | $0\to1$ | $u+\kappa s$ | 随机 | $\rho(t)$ |
| 反向 SDE | $1\to0$ | $u-\kappa s$，配合负的 $\mathrm dt$ | 随机 | 反向经过同一组 $\rho(t)$ |

需要强调的是：

$$
\boxed{
\text{三种动力学的单条轨迹通常不同；相同的是相应时刻的边际密度。}
}
$$

本章的核心结论可以压缩为

$$
\boxed{
\text{布朗扩散}
\mathbin{+}
\text{score 漂移补偿}
\mathrel{=}
\text{不改变目标边际概率流}.
}
$$
## 7. 平方回归：为什么随机标签可以学习出确定速度场

### 7.1 条件期望是平方损失的最优预测

令神经网络 $u_\theta(t,x)$ 逼近速度场。考虑均方损失

$$
\boxed{
\mathcal J_{\mathrm{vel}}(\theta)
\mathrel{=}
\mathbb E_{t,x_0,x_1,z}
\left[
\left\|
u_\theta(t,x_t)-Y_t
\right\|^2
\right].
}
\tag{7.1}
$$

其中通常取

$$
t\sim\mathrm{Unif}(0,1),
$$

并按式 (2.1) 构造

$$
x_t=I(t,x_0,x_1)+\gamma(t)z,
$$

按式 (3.1) 构造随机速度标签

$$
Y_t
\mathrel{=}
\partial_tI(t,x_0,x_1)+\dot\gamma(t)z.
$$

对固定的 $t$，记

$$
m_t(x)
:=
\mathbb E[Y_t\mid x_t=x],
\qquad
R_t
:=
Y_t-m_t(x_t).
$$

$m_t(x)$ 是给定位置后的条件平均速度，实际上就是

$$
m_t(x)=u(t,x).
$$

$R_t$ 是单个随机速度标签相对于条件平均的剩余项。由条件期望的定义，

$$
\boxed{
\mathbb E[R_t\mid x_t]=0.
}
$$

也就是说：在固定 $x_t$ 后，剩余项 $R_t$ 的平均为零。

对任意候选预测器 $f(t,x_t)$，将

$$
Y_t=m_t(x_t)+R_t
$$

代入平方误差：

$$
\begin{aligned}
\|f(t,x_t)-Y_t\|^2
&=
\|f(t,x_t)-m_t(x_t)-R_t\|^2
\\
&=
\|f(t,x_t)-m_t(x_t)\|^2
\mathbin{+}
\|R_t\|^2
\\
&\quad
-2\bigl(f(t,x_t)-m_t(x_t)\bigr)\cdot R_t.
\end{aligned}
$$

对上式取期望。交叉项的期望为零，因为

$$
\begin{aligned}
&\mathbb E[
\bigl(f(t,x_t)-m_t(x_t)\bigr)\cdot R_t
]
\\
&=
\mathbb E\!\left[
\mathbb E[
\bigl(f(t,x_t)-m_t(x_t)\bigr)\cdot R_t
\mid x_t]
\right]
\\
&=
\mathbb E\!\left[
\bigl(f(t,x_t)-m_t(x_t)\bigr)\cdot
\mathbb E[R_t\mid x_t]
\right]
\\
&=0.
\end{aligned}
$$

第二个等号使用了：给定 $x_t$ 后，$f(t,x_t)-m_t(x_t)$ 已经是确定量，因此可以移到内层条件期望之外。

于是得到正交分解恒等式

$$
\boxed{
\begin{aligned}
\mathbb E\|f(t,x_t)-Y_t\|^2
&=
\mathbb E\left\|
f(t,x_t)-m_t(x_t)
\right\|^2
\\
&\quad+
\mathbb E\left\|
Y_t-m_t(x_t)
\right\|^2.
\end{aligned}
}
\tag{7.2}
$$

右侧两项的含义是：

1. 第一项

   $$
   \mathbb E\|f(t,x_t)-m_t(x_t)\|^2
   $$

   表示预测器 $f$ 与条件平均速度之间的误差，它可以通过选择 $f$ 来减小。

2. 第二项

   $$
   \mathbb E\|Y_t-m_t(x_t)\|^2
   $$

   是隐藏端点和高斯潜变量造成的不可约条件方差，它与预测器 $f$ 无关。

因此，右侧唯一能通过 $f$ 改变的是第一项。它在

$$
f(t,x)=m_t(x)
$$

时达到最小值零。令 $f=u_\theta$，再对时间 $t$ 取期望，得到总体最优解

$$
\boxed{
u_\theta^*(t,x)
\mathrel{=}
\mathbb E[Y_t\mid x_t=x]
\mathrel{=}
u(t,x).
}
\tag{7.3}
$$

这正是“带隐藏变量的随机监督标签为什么仍能训练出确定速度场”的原因。单个标签 $Y_t$ 可以有很大方差，但网络只观察 $(t,x_t)$；平方回归会把同一 $(t,x)$ 对应的隐藏端点与噪声平均掉，输出条件均值 $u(t,x)$。

### 7.2 与论文二次目标的关系

参考论文使用

$$
\boxed{
\mathcal L_u[\hat u]
\mathrel{=}
\int_0^1
\mathbb E\left[
\frac12\|\hat u(t,x_t)\|^2
-Y_t\cdot\hat u(t,x_t)
\right]\mathrm dt.
}
\tag{7.4}
$$

因为

$$
\frac12\|\hat u-Y_t\|^2
\mathrel{=}
\frac12\|\hat u\|^2
-Y_t\cdot\hat u
\mathbin{+}
\frac12\|Y_t\|^2,
$$

所以式 (7.4) 与均方损失只相差

$$
\frac12\mathbb E\|Y_t\|^2,
$$

这一项不依赖网络 $\hat u$。因此两个目标函数具有完全相同的最优解。

### 7.3 实际训练步骤

一次无偏的随机训练迭代可以写成：

1. 采样 $t\sim\mathrm{Unif}(0,1)$；
2. 采样 $(x_0,x_1)\sim\nu$；
3. 采样 $z\sim\mathcal N(0,\mathrm{Id})$；
4. 构造

   $$
   x_t=I(t,x_0,x_1)+\gamma(t)z;
   $$

5. 构造监督标签

   $$
   Y_t=\partial_tI(t,x_0,x_1)+\dot\gamma(t)z;
   $$

6. 最小化

   $$
   \|u_\theta(t,x_t)-Y_t\|^2.
   $$

训练阶段只需要从端点分布和高斯分布采样，不需要知道中间密度 $\rho(t,x)$，也不需要显式计算

$$
\mathbb E[Y_t\mid x_t=x].
$$

条件期望由平方回归自动实现。

---

## 8. Flow Matching 是上述框架的一个特例

### 8.1 去掉额外高斯潜变量

令

$$
\gamma(t)=0,
\qquad
x_t=I(t,x_0,x_1).
$$

则

$$
Y_t=\partial_tI(t,x_0,x_1),
$$

速度场为

$$
\boxed{
u(t,x)
\mathrel{=}
\mathbb E[
\partial_tI(t,x_0,x_1)
\mid
I(t,x_0,x_1)=x
].
}
\tag{8.1}
$$

训练目标为

$$
\boxed{
\mathcal L_{\mathrm{FM}}(\theta)
\mathrel{=}
\mathbb E
\left\|
u_\theta(t,I(t,x_0,x_1))
\mathbin{-}
\partial_tI(t,x_0,x_1)
\right\|^2.
}
\tag{8.2}
$$

这是端点插值形式的 Flow Matching。更一般的 Conditional Flow Matching 也可以从条件概率路径出发；这里讨论的是最直接、最常见的一种表示。

### 8.2 线性插值与 Rectified Flow

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

因此

$$
\boxed{
\mathcal L_{\mathrm{linear\text{-}FM}}
\mathrel{=}
\mathbb E
\left\|
u_\theta\bigl(t,(1-t)x_0+tx_1\bigr)
\mathbin{-}
(x_1-x_0)
\right\|^2.
}
\tag{8.3}
$$

这也是 Rectified Flow 的基础训练形式。

即使每条条件路径都是直线，条件平均后的边际速度场 $u(t,x)$ 也不必产生直的 ODE 轨迹。端点耦合会影响条件速度的方差、边际轨迹的弯曲程度和学习难度。

---

## 9. 从高斯潜变量得到 score

回到双端点随机插值

$$
x_t=I(t,x_0,x_1)+\gamma(t)z.
$$

对满足 $\gamma(t)>0$ 的内部时刻，给定

$$
m=I(t,x_0,x_1)
$$

后，$x_t$ 是均值为 $m$、协方差为 $\gamma^2(t)\mathrm{Id}$ 的高斯变量。高斯分部积分给出

$$
\boxed{
s(t,x)
\mathrel{=}
\nabla_x\log\rho(t,x)
\mathrel{=}
-\frac{1}{\gamma(t)}
\mathbb E[z\mid x_t=x].
}
\tag{9.1}
$$

定义噪声预测器

$$
\eta_z(t,x)
\mathrel{=}
\mathbb E[z\mid x_t=x],
$$

则

$$
\boxed{
s(t,x)=-\frac{\eta_z(t,x)}{\gamma(t)}.
}
\tag{9.2}
$$

它可以通过平方回归学习：

$$
\mathcal L_{\mathrm{noise}}
\mathrel{=}
\mathbb E
\left\|
\eta_\theta(t,x_t)-z
\right\|^2.
\tag{9.3}
$$

另外定义

$$
v(t,x)
\mathrel{=}
\mathbb E[
\partial_tI(t,x_0,x_1)
\mid x_t=x
].
$$

由

$$
\mathbb E[z\mid x_t=x]
\mathrel{=}
-\gamma(t)s(t,x)
$$

得到

$$
\boxed{
u(t,x)
\mathrel{=}
v(t,x)-\dot\gamma(t)\gamma(t)s(t,x).
}
\tag{9.4}
$$

这条公式连接了速度预测、噪声预测和 score 预测。

---

## 10. 扩散模型使用的是一侧高斯加噪路径

这一节需要与第 2 节的“双端点随机插值”明确区分。

### 10.1 加噪时间约定

扩散模型通常使用加噪时间 $\tau$：

$$
\tau=0\text{ 是数据},
\qquad
\tau=T\text{ 是近似高斯噪声}.
$$

设

$$
y_0\sim p_{\mathrm{data}},
\qquad
\varepsilon\sim\mathcal N(0,\mathrm{Id}),
$$

并定义一侧高斯加噪路径

$$
\boxed{
y_\tau
\mathrel{=}
a(\tau)y_0+\sigma(\tau)\varepsilon.
}
\tag{10.1}
$$

通常

$$
a(0)=1,
\qquad
\sigma(0)=0,
$$

而在终点

$$
a(T)\approx0,
\qquad
\sigma(T)\approx1.
$$

这是一侧高斯 corruption path。它可以在时间反向后把 $\varepsilon$ 看作高斯基分布端点，也可以视为放宽端点条件后的 one-sided stochastic interpolant。

它不应在没有说明的情况下与第 2 节满足

$$
\gamma(0)=\gamma(1)=0
$$

的额外潜变量直接混为同一个定义。

### 10.2 直接推导噪声预测与 score 的关系

记 $y_\tau$ 的密度为 $r_\tau(y)$，以免与第 6 节 SDE 候选过程的密度 $p(t,x)$ 混淆。给定 $y_0$ 后，

$$
y_\tau\mid y_0
\sim
\mathcal N(a(\tau)y_0,\sigma^2(\tau)\mathrm{Id}).
$$

对高斯混合密度求梯度可得

$$
\begin{aligned}
\nabla_y\log r_\tau(y)
&=
\mathbb E\left[
\frac{a(\tau)y_0-y}{\sigma^2(\tau)}
\;\middle|\;
y_\tau=y
\right]
\\
&=
-\frac{1}{\sigma(\tau)}
\mathbb E[
\varepsilon\mid y_\tau=y
].
\end{aligned}
$$

因此

$$
\boxed{
\nabla_y\log r_\tau(y)
\mathrel{=}
-\frac{
\mathbb E[\varepsilon\mid y_\tau=y]
}{
\sigma(\tau)
}.
}
\tag{10.2}
$$

式 (10.2) 与式 (9.1) 同型，但这里是针对一侧高斯加噪模型直接推导的，不依赖第 2 节的双端点条件。

---

## 11. 连续时间 VP 扩散、反向 SDE 与概率流 ODE

### 11.1 一般前向加噪 SDE

考虑

$$
\boxed{
\mathrm dD_\tau
\mathrel{=}
f(\tau,D_\tau)\,\mathrm d\tau
\mathbin{+}
g(\tau)\,\mathrm dW_\tau.
}
\tag{11.1}
$$

其密度 $r_\tau$ 满足

$$
\partial_\tau r_\tau
\mathrel{=}
-\nabla\cdot(fr_\tau)
\mathbin{+}
\frac12g^2(\tau)\Delta r_\tau.
\tag{11.2}
$$

令

$$
s_\tau(y)=\nabla_y\log r_\tau(y).
$$

因为

$$
\Delta r_\tau
\mathrel{=}
\nabla\cdot(r_\tau s_\tau),
$$

式 (11.2) 可以改写为连续性方程

$$
\partial_\tau r_\tau
\mathrel{=}
-\nabla\cdot
\left[
\left(
f-\frac12g^2s_\tau
\right)r_\tau
\right].
$$

所以概率流 ODE 是

$$
\boxed{
\mathrm dD_\tau
\mathrel{=}
\left[
f(\tau,D_\tau)
\mathbin{-}
\frac12g^2(\tau)s_\tau(D_\tau)
\right]\mathrm d\tau.
}
\tag{11.3}
$$

它与前向加噪 SDE 具有相同的单时刻边际。

从 $\tau=T$ 向 $\tau=0$ 积分的反向时间 SDE 是

$$
\boxed{
\mathrm dD_\tau
\mathrel{=}
\left[
f(\tau,D_\tau)
\mathbin{-}
g^2(\tau)s_\tau(D_\tau)
\right]\mathrm d\tau
\mathbin{+}
g(\tau)\,\mathrm d\overline W_\tau,
\qquad
\mathrm d\tau<0.
}
\tag{11.4}
$$

### 11.2 VP SDE

Variance-Preserving SDE 取

$$
f(\tau,y)
\mathrel{=}
-\frac12\beta(\tau)y,
\qquad
g(\tau)=\sqrt{\beta(\tau)}.
$$

于是

$$
\boxed{
\mathrm dD_\tau
\mathrel{=}
-\frac12\beta(\tau)D_\tau\,\mathrm d\tau
\mathbin{+}
\sqrt{\beta(\tau)}\,\mathrm dW_\tau.
}
\tag{11.5}
$$

定义

$$
\bar\alpha(\tau)
\mathrel{=}
\exp\left(
-\int_0^\tau\beta(r)\,\mathrm dr
\right),
$$

则条件边际为

$$
\boxed{
D_\tau
\mathrel{=}
\sqrt{\bar\alpha(\tau)}D_0
\mathbin{+}
\sqrt{1-\bar\alpha(\tau)}\varepsilon.
}
\tag{11.6}
$$

这正是式 (10.1) 的特殊情况：

$$
a(\tau)=\sqrt{\bar\alpha(\tau)},
\qquad
\sigma(\tau)=\sqrt{1-\bar\alpha(\tau)}.
$$

![前向扩散的边际逐步高斯化](assets/unified-diffusion/forward-diffusion-marginals.png)

*图 2：前向扩散逐步平滑数据密度并使其接近高斯先验。有限 $T$ 时通常只是近似高斯；只有在 $\bar\alpha(T)=0$ 的理想极限下才精确成为标准高斯。图源：Nakkiran 等，Step-by-Step Diffusion, Figure 1。*

---

## 12. DDPM：离散 VP 加噪链

DDPM 首先定义离散前向马尔可夫链，而不是直接把任意随机插值公式做一次普通离散化。

### 12.1 前向链与闭式边际

记数据为 $y_0$。定义

$$
q_{\mathrm{fwd}}(y_k\mid y_{k-1})
\mathrel{=}
\mathcal N\left(
\sqrt{\alpha_k}y_{k-1},
(1-\alpha_k)\mathrm{Id}
\right),
\tag{12.1}
$$

其中

$$
\beta_k:=1-\alpha_k,
\qquad
\bar\alpha_k:=\prod_{j=1}^k\alpha_j.
$$

将前 $k$ 步噪声合并：

$$
\boxed{
y_k
\mathrel{=}
\sqrt{\bar\alpha_k}y_0
\mathbin{+}
\sqrt{1-\bar\alpha_k}\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,\mathrm{Id}).
}
\tag{12.2}
$$

它是连续 VP 条件边际式 (11.6) 的离散对应。

这里用 $q_{\mathrm{fwd}}$ 表示 DDPM 的前向加噪分布，以免与第 5 节的 ODE 密度 $q(t,x)$ 混淆。

有限 $K$ 时，$q_{\mathrm{fwd}}(y_K)$ 通常只是接近 $\mathcal N(0,\mathrm{Id})$。常见采样算法仍以标准高斯作为近似终端先验。

### 12.2 噪声预测就是尺度化的 score 预测

对式 (12.2) 直接使用第 10.2 节的一侧高斯恒等式：

$$
\boxed{
\nabla_y\log q_{\mathrm{fwd},k}(y)
\mathrel{=}
-\frac{
\mathbb E[\varepsilon\mid y_k=y]
}{
\sqrt{1-\bar\alpha_k}
}.
}
\tag{12.3}
$$

训练噪声网络

$$
\boxed{
\mathcal L_{\mathrm{DDPM}}
\mathrel{=}
\mathbb E
\left\|
\varepsilon_\theta(y_k,k)-\varepsilon
\right\|^2.
}
\tag{12.4}
$$

平方损失的理想最优解为

$$
\varepsilon_\theta^*(y,k)
\mathrel{=}
\mathbb E[\varepsilon\mid y_k=y].
$$

所以对应的 score 估计是

$$
\boxed{
s_\theta(y_k,k)
\mathrel{=}
-\frac{
\varepsilon_\theta(y_k,k)
}{
\sqrt{1-\bar\alpha_k}
}.
}
\tag{12.5}
$$

网络同时给出干净数据估计

$$
\boxed{
\widehat y_0
\mathrel{=}
\frac{
y_k-\sqrt{1-\bar\alpha_k}\varepsilon_\theta(y_k,k)
}{
\sqrt{\bar\alpha_k}
}.
}
\tag{12.6}
$$

### 12.3 精确条件后验与模型反向更新

给定 $y_0$ 时，精确后验为

$$
q_{\mathrm{fwd}}(y_{k-1}\mid y_k,y_0)
\mathrel{=}
\mathcal N(
\widetilde\mu_k,
\widetilde\beta_k\mathrm{Id}
),
\tag{12.7}
$$

其中

$$
\widetilde\beta_k
\mathrel{=}
\frac{1-\bar\alpha_{k-1}}
{1-\bar\alpha_k}\beta_k,
\tag{12.8}
$$

$$
\widetilde\mu_k(y_k,y_0)
\mathrel{=}
\frac{
\sqrt{\bar\alpha_{k-1}}\beta_k
}{
1-\bar\alpha_k
}y_0
\mathbin{+}
\frac{
\sqrt{\alpha_k}(1-\bar\alpha_{k-1})
}{
1-\bar\alpha_k
}y_k.
\tag{12.9}
$$

用网络预测替代未知 $y_0$ 后，常用均值参数化为

$$
\boxed{
\mu_\theta(y_k,k)
\mathrel{=}
\frac1{\sqrt{\alpha_k}}
\left(
y_k
\mathbin{-}
\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\varepsilon_\theta(y_k,k)
\right).
}
\tag{12.10}
$$

DDPM 反向采样为

$$
\boxed{
y_{k-1}
\mathrel{=}
\mu_\theta(y_k,k)
\mathbin{+}
\sigma_k\xi_k,
\qquad
\xi_k\sim\mathcal N(0,\mathrm{Id}).
}
\tag{12.11}
$$

常见选择之一是

$$
\sigma_k^2=\widetilde\beta_k.
$$

需要区分：

- $q_{\mathrm{fwd}}(y_{k-1}\mid y_k,y_0)$ 是精确高斯后验；
- 边缘化 $y_0$ 后的 $q_{\mathrm{fwd}}(y_{k-1}\mid y_k)$ 一般是高斯混合，不必精确为单个高斯；
- $p_\theta(y_{k-1}\mid y_k)$ 是模型使用的高斯反向核。

在小步长、连续时间极限下，DDPM 的离散反向过程与 VP 反向 SDE 对应。有限步 DDPM 更准确地说是离散马尔可夫链的学习式反转，而不是任意连续 ODE/SDE 的精确一步离散化。

---

## 13. DDIM：共享扰动边际和预测器，改变采样耦合

DDIM 继续使用与 DDPM 相同的训练边际

$$
y_k
\mathrel{=}
\sqrt{\bar\alpha_k}y_0
\mathbin{+}
\sqrt{1-\bar\alpha_k}\varepsilon
$$

以及同一个 $\varepsilon_\theta$，但采用不同的跨时刻耦合和反向更新。

### 13.1 确定性 DDIM

先计算

$$
\widehat y_0
\mathrel{=}
\frac{
y_k-\sqrt{1-\bar\alpha_k}\varepsilon_\theta(y_k,k)
}{
\sqrt{\bar\alpha_k}
}.
$$

然后使用

$$
\boxed{
y_{k-1}
\mathrel{=}
\sqrt{\bar\alpha_{k-1}}\widehat y_0
\mathbin{+}
\sqrt{1-\bar\alpha_{k-1}}
\varepsilon_\theta(y_k,k).
}
\tag{13.1}
$$

给定 $y_k$ 和模型输出后，式 (13.1) 不再注入新的随机噪声，因此是确定性更新。

### 13.2 一般的随机性参数

更一般地，

$$
\boxed{
\begin{aligned}
y_{k-1}
={}&
\sqrt{\bar\alpha_{k-1}}\widehat y_0
\\
&+
\sqrt{
1-\bar\alpha_{k-1}-\sigma_k^2
}
\varepsilon_\theta(y_k,k)
\mathbin{+}
\sigma_k\xi_k,
\end{aligned}
}
\tag{13.2}
$$

其中

$$
\boxed{
\sigma_k
\mathrel{=}
\eta_{\mathrm{DDIM}}
\sqrt{
\frac{1-\bar\alpha_{k-1}}
{1-\bar\alpha_k}
}
\sqrt{
1-\frac{\bar\alpha_k}{\bar\alpha_{k-1}}
}.
}
\tag{13.3}
$$

- $\eta_{\mathrm{DDIM}}=0$：确定性 DDIM；
- $\eta_{\mathrm{DDIM}}=1$：$\sigma_k^2=\widetilde\beta_k$，对应常用 DDPM 方差选择；
- 中间取值：在确定性与随机性之间插值。

在适当时间参数化和连续时间极限下，确定性 DDIM 与概率流 ODE 密切对应。有限步 DDIM 是一种特定离散传输更新，不应把每一步都理解为概率流 ODE 的精确解。

![DDPM、DDIM、反向 SDE 与概率流 ODE 的关系](assets/unified-diffusion/reverse-samplers-concept-map.png)

*图 3：离散/连续与随机/确定性两个维度下的概念关系。图中使用原论文自己的符号和时间方向。图源：Nakkiran 等，Step-by-Step Diffusion, Figure 6。*

### 13.3 “共享边际”应如何准确理解

DDPM 与 DDIM 确实共享：

1. 训练时的一侧高斯扰动边际 $q_{\mathrm{fwd}}(y_k\mid y_0)$；
2. 噪声预测器 $\varepsilon_\theta$；
3. 对应的 score 参数化；
4. 干净数据估计 $\widehat y_0$。

但在有限步、近似模型下，实际生成过程的中间边际不必精确相同。所谓“共享边际”首先指它们共享训练时规定的前向扰动边际和理想化构造。

---

## 14. 最终统一视角

### 14.1 统一对象

| 方法 | 先规定什么 | 学习什么 | 生成方式 |
|---|---|---|---|
| Stochastic Interpolant | 双端点随机插值边际 $\rho(t)$ | 条件平均速度、score | ODE 或同边际 SDE |
| Flow Matching | 条件概率路径或端点插值 | 条件速度的边际平均 | ODE |
| Rectified Flow | 线性端点插值 | $x_1-x_0$ 的条件平均 | ODE |
| Score-based diffusion | 一侧高斯加噪边际 | score 或等价噪声 | 反向 SDE / 概率流 ODE |
| DDPM | 离散 VP 加噪链 | 噪声或等价 score | 随机反向马尔可夫链 |
| DDIM | 与 DDPM 相同的训练扰动边际 | 同一个预测器 | 确定性或部分随机更新 |

### 14.2 最关键的两条证明链

第一条是 ODE：

$$
\boxed{
\begin{aligned}
x_t\sim\rho
&\Longrightarrow
u=\mathbb E[\dot x_t\mid x_t=x],
\\
\partial_t\rho+\nabla\cdot(\rho u)&=0,
\\
\dot X_t=u(t,X_t),\quad X_t\sim q
&\Longrightarrow
\partial_tq+\nabla\cdot(qu)=0,
\\
\text{同一 PDE}+\text{同一初值}+\text{唯一性}
&\Longrightarrow
q=\rho.
\end{aligned}
}
\tag{14.1}
$$

第二条是 SDE：

$$
\boxed{
\begin{aligned}
s&=\nabla\log\rho,
\\
\mathrm dZ_t
&=
(u+\kappa s)(t,Z_t)\,\mathrm dt
\mathbin{+}
\sqrt{2\kappa(t)}\,\mathrm dW_t,
\\
\rho s&=\nabla\rho
\\
&\Longrightarrow
\text{$\rho$ 满足该 SDE 的 Fokker--Planck 方程}
\\
&\Longrightarrow
p=\rho.
\end{aligned}
}
\tag{14.2}
$$

### 14.3 一句话总结

$$
\boxed{
\begin{gathered}
\text{随机插值负责设计和监督一条概率路径，}
\\
\text{Flow Matching 学习实现该路径的条件平均速度，}
\\
\text{score 决定如何在不改变边际的情况下加入或反转随机扩散，}
\\
\text{DDPM 与 DDIM 是一侧高斯扰动路径下的离散随机和确定性实现。}
\end{gathered}
}
$$

ODE、SDE、DDPM 和 DDIM 的轨迹结构不同，有限步近似也不同；统一之处主要在于它们可以围绕同一组随时间变化的边际分布、同一类条件期望回归以及相互转换的速度/score 参数化来理解。

---

## 参考文献

1. Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, [*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://arxiv.org/abs/2303.08797), arXiv:2303.08797v4.
2. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747), ICLR 2023.
3. Xingchao Liu, Chengyue Gong, Qiang Liu, [*Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*](https://arxiv.org/abs/2209.03003), ICLR 2023.
4. Jonathan Ho, Ajay Jain, Pieter Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), NeurIPS 2020.
5. Jiaming Song, Chenlin Meng, Stefano Ermon, [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502), ICLR 2021.
6. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole, [*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456), ICLR 2021.
7. Preetum Nakkiran, Arwen Bradley, Hattie Zhou, Madhu Advani, [*Step-by-Step Diffusion: An Elementary Tutorial*](https://arxiv.org/abs/2406.08929), arXiv:2406.08929.
