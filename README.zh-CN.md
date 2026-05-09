# New-IL

New-IL 是一个关于 **Progress-Aware Trajectory Cloud Supervision** 的 idea 存档，核心关注事件约束模仿学习中的动作监督目标。

> 这不是严谨的学术发表，而是一个用于保存、讨论、启发和寻找潜在合作者的探索性存档。

## 核心直觉

很多 imitation learning 方法会把 action chunk 里的每一个未来动作，都监督到 demonstration 中某个固定时间点上。但机器人操作里，这样太僵硬。

连续运动阶段应该允许弹性：

> 可以稍微快一点、慢一点、提前一点、滞后一点，或者沿着略有差异的路径走，只要仍然处在当前阶段合理的轨迹云管道内。

但关键事件必须精确：

> 夹爪闭合、夹爪打开、接触、释放、插入、阶段切换，不能被云化、平均化或模糊化。

所以动作监督既不应该统一写成固定时间点回归，也不应该把所有动作维度都统一分布化。它应该尊重机器人操作中不同时间片段的异质语义。

## 核心公式

传统监督通常强迫：

$$
\hat{x}_{t+k} \rightarrow x^\star_{t+k}
$$

PA-TCS 则允许预测动作在一个进度区间内匹配轨迹云：

$$
d_{\mathrm{tube}}(\hat{x}_{t+k})
=
\min_{\rho\in I_{t,k}}
d_{\mathcal{C}_r}(\hat{x}_{t+k},\rho)
$$

其中 $I_{t,k}$ 是允许的进度区间，$\mathcal{C}_r$ 是第 $r$ 个阶段的轨迹云。

完整目标把阶段内的进度弹性监督和关键事件的精确约束结合起来：

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{tube}}
+
\lambda_e\mathcal{L}_{\mathrm{event}}
+
\lambda_c\mathcal{L}_{\mathrm{cross}}
+
\lambda_m\mathcal{L}_{\mathrm{mono}}
+
\lambda_v\mathcal{L}_{\mathrm{speed}}
$$

## 这个想法有趣在哪里

- 它把“进度误差”和“真正偏离轨迹”区分开。
- 它保留连续动作的多解性，但不牺牲关键事件精度。
- 它给 action chunk 一个阶段感知的监督结构。
- 它自然引出比 success rate 更细的指标，例如 tube violation rate、event timing error、event pose error、wrong crossing rate 和 progress backward rate。

## 文档

- 英文：`docs/en/readme.md`
- 中文：`docs/zh-CN/readme.md`

## License

本项目使用 MIT License。详见 `LICENSE`。
