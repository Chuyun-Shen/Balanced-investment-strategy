# 智能投资再平衡助手（小白友好）

让普通投资者也能像专业投顾一样，**按计划、可落地**地打理资产配置。你只需要设定目标权重，程序会**每周**自动给出具体的买入/减持清单。

---

## 它解决了哪些痛点？

* **情绪干扰** ：下跌恐慌、上涨贪婪？交给规则；按纪律走。
* **执行困难** ：知道该调仓，但总拖延？每周给出**清单级**建议。
* **资金节奏** ：一次投太多/太少？按偏离度与风险，智能**分批**推进。

> 再平衡的核心，是把偏离目标配置的组合 **拉回正轨** ，帮助维持既定风险水平并促成“低卖高买”的纪律化执行。权威机构（如 Vanguard、Bogleheads 社区）均将再平衡视为长期成功的重要实践。([investor.vanguard.com](https://investor.vanguard.com/investor-resources-education/portfolio-management/rebalancing-your-portfolio?utm_source=chatgpt.com "Rebalancing your portfolio: How to rebalance - Vanguard"), [理财头脑](https://www.bogleheads.org/wiki/Rebalancing?utm_source=chatgpt.com "Rebalancing"))

---

## 开箱即用：一步步来

### 1) 安装依赖

```bash
pip install akshare pandas numpy yfinance pandas-datareader requests beautifulsoup4
```

### 2) 配置当前持仓与目标

* 实现config/custom_positions.py
  ```python
  current_positions = {
    "161119": 10000,
    "510300": 10000,
    "009051": 10000,
    # ... 你的其它持仓金额
  }
  ```
* 目标权重在 `rebalance_helper.py` 顶部的 `targets` 字典中（可直接按你给的默认方案使用或微调）。

### 3) 运行看看本周建议

```bash
# 只打印（不导出 CSV）
python -m rebalance_helper --no-export --plan B

# 控制单次交易比例、选择方案、是否把被削减的权益回流到债/红利
python -m rebalance_helper --no-export --max-trade-pct 0.3 --plan B --realloc none
```

* **方案A** ：考虑场内 ETF 100 股整手；减持优先动开放式/LOF
* **方案B** ：把所有标的当作"可任意金额"处理（更平滑、适合纯基金）

> **提示**：您可以根据自己的实际情况灵活替换标的，例如用场外指数基金替代对应的ETF，只要保持资产类别和风险特征一致即可。

---

## 默认资产配置（适合多数中等风险投资者）

| 类别      | 占比 | 作用                       |
| --------- | ---: | -------------------------- |
| 防守资产  |  35% | 降波动、保稳健             |
| 核心权益  |  40% | 长期增长的主力             |
| 海外权益  |  15% | 全球分散，降低单一市场风险 |
| 商品/替代 |  10% | 抗通胀、改善尾部风险       |

> 这套 65/35 左右的股债结构，兼顾增长与稳定；你也可以按自身偏好调整，但建议始终**保持分散**与 **长期** 。关于"定投/分批投入"是否优于一次性投入，Vanguard 的研究指出：长期看一次性投入的胜率略高，但 **分批更容易帮助投资者坚持计划、降低后悔情绪** ，对许多投资者反而更友好。([investor.vanguard.com](https://investor.vanguard.com/investor-resources-education/news/lump-sum-investing-versus-cost-averaging-which-is-better?utm_source=chatgpt.com "Lump-sum investing versus cost averaging: Which is better?"), [先锋投资](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf?utm_source=chatgpt.com "Cost averaging: Invest now or temporarily hold your cash?"))

 **标的与权重明细** （可直接使用）：

```
# 防守资产 35%
161119  25%   | 易方达中债新综债LOF（人民币债，久期基石）
007360  10%   | 易方达中短期美元债(QDII)A（短久期IG，低波票息）

# 核心权益 40%
510300  15%   | 沪深300ETF（核心宽基）
009051  15%   | 中证红利ETF/联接（股息/质量、回撤缓冲）
510500   6%   | 中证500ETF（中盘补充）
001917   4%   | 招商量化精选A（阿尔法补充）

# 海外权益 15%
513500  10%   | 标普500ETF
513100   2%   | 纳指100ETF（成长卫星，估值克制）
513800   3%   | 日本东证ETF（治理红利）

# 商品/替代 10%
518880   6%   | 黄金ETF（通胀与尾部对冲）
588000   2%   | 科创50ETF（谨慎参与）
164824   2%   | 印度LOF（新兴市场卫星）
```

> **资产选择灵活性提示**：上述标的仅为推荐，您完全可以根据自己的实际情况进行替换。例如，场内ETF（如510300沪深300ETF）可以替换为对应的场外指数基金（如000311易方达沪深300指数）；同样，场外基金也可以替换为对应的场内ETF。关键是保持资产类别和风险特征的一致性，而非严格遵循特定代码。

---

## 它如何做决策？（一眼能懂的流程）

1. **测偏离** ：比较“当前持仓” vs “目标金额”，确定需要补的标的与规模。
2. **带宽过滤** ：轻微偏离不动（默认：绝对 1.5 个百分点；相对 ±20% 带宽），减少“高位小额补仓”。
3. **强度引擎** ：用三个维度给“本周资金”定速——

* **估值** ：如沪深300 PE、股息率相对 10 年国债的“收益率缺口”等；贵时降速、便宜时加速。
* **价量动量 + 位阶** ：如 3/12 个月动量、MA50 斜率、量能比、距离 52 周高/MA200 乖离的“位阶惩罚”。
* **风险挡板** ：看组合自身的 **月度回撤** 、 **年化波动** ，动态限制**权益上限**与出手强度。

1. **分批额度** ：结合“与目标差额”“最大单次比例”“现金储备”“偏离度”算出当周总额度。
2. **下单分配** ：按**缺口占比**分配到各标的；触发器会做必要调整（如黄金不追、纳指上限、整手回流等）。
3. **两套方案** ：A（整手/减持动基金）与 B（纯基金），任选更契合你账户的方案。

> 对“多久再平衡一次”与“阈值再平衡是否更优”，Vanguard 的多份研究与实践总结指出：**过于频繁或过于稀疏**都不理想；对很多投资者而言， **年度或阈值式再平衡** （如超出带宽再动）更兼顾成本、跟踪误差与执行度。([先锋投资](https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/tuning-frequency-for-rebalancing.html?utm_source=chatgpt.com "Finding the optimal rebalancing frequency | Vanguard"))

---

## 如何读懂程序输出？

* **CSI300 PE/E-P** ：估值与盈利收益率（E/P）。E/P 越高，相对债券的“收益率缺口”越大，一般更有利于权益配置。
* **CN10Y/美 IG 1–3Y** ：国内 10 年国债与美元短久期 IG 债收益率；用于衡量**无风险回报**与跨市场机会。
* 本工具的 CN10Y 来自**中国外汇交易中心（CFETS/中国货币网）或中债**公开页面解析与回退，稳定可用。([shibor.org](https://www.shibor.org/english/bmkierirt/?utm_source=chatgpt.com "Interest Rate (SDDS) - CFETS"), [ChinaBond Yield Curves](https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/more?locale=en_US&utm_source=chatgpt.com "CGB Yield Curve and Others - ChinaBond"))
* **动量/量能/位阶** ：反映市场趋势与位置；当“位阶分位”极高时会触发 **位阶惩罚** ，自动收敛强度。
* **组合月度回撤/年化波动/权益上限** ：当回撤扩大或波动偏高，系统会 **降低强度并压住权益比例** ，守住底线。
* **方案 A/B 清单** ：逐标的本周 **建议买入/减持金额** ，以及交易前后配比的变化。

---

## 我该怎么调自己的目标权重？

* **更保守** ：把防守资产提升到 50%+，权益降到 50% 以下。
* **更积极** ：权益到 70%–80%，但建议**保留 ≥20% 防守**以应对大回撤。
* **按人生阶段** ：
* 积累期：权益占比可更高，追求长期增长；
* 稳定期：均衡配置；
* 分配期：提高防守比例，降低波动与回撤。

---

## 常见问题（FAQ）

**Q1：为什么“本周只买了债/红利”？**

A：很可能是**带宽过滤** + **位阶/估值/风险挡板**共同作用：当权益已接近位阶上沿或风险指标发出“降速”信号时，系统会优先把增量投向 **防守资产** ，以保持组合整体的风险回报结构。

**Q2：我想"极简"，能否只用方案 B（纯基金）？是否必须使用推荐的标的？**

A：可以。方案 B 会把整手/卖出约束一并简化，适合以开放式基金为主的账户；场内 ETF 也能按金额拆分执行（以目标配比为主）。您可以根据自己的实际情况，用相同风险特征的其他基金替代推荐标的，例如用场外指数基金替代ETF，或者用相同指数的不同基金公司产品替代。重要的是保持资产类别和风险特征的一致性，而非严格遵循特定代码。

**Q3：分批是不是一定优于一次性？**

A：长期统计上，**一次性**投入的胜率略高，但分批更能帮多数投资者 **坚持计划、降低后悔** ，特别是在波动或情绪压力较大的阶段。选择你更能长期坚持的方式更重要。([investor.vanguard.com](https://investor.vanguard.com/investor-resources-education/news/lump-sum-investing-versus-cost-averaging-which-is-better?utm_source=chatgpt.com "Lump-sum investing versus cost averaging: Which is better?"), [先锋投资](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf?utm_source=chatgpt.com "Cost averaging: Invest now or temporarily hold your cash?"))

**Q4：再平衡会不会增加交易成本？**

A：本工具用了 **带宽阈值** 、 **整手回流** 、**优先动基金**等机制，尽量减少频繁、零碎的交易；实际仍需结合你的券商费率与税费来考量。

---

## 数据来源与致谢

* **再平衡理念与实践** ：Vanguard 投资者教育与研究报告；Bogleheads 社区知识库。([investor.vanguard.com](https://investor.vanguard.com/investor-resources-education/portfolio-management/rebalancing-your-portfolio?utm_source=chatgpt.com "Rebalancing your portfolio: How to rebalance - Vanguard"), [先锋投资](https://corporate.vanguard.com/content/corporatesite/us/en/corp/articles/tuning-frequency-for-rebalancing.html?utm_source=chatgpt.com "Finding the optimal rebalancing frequency | Vanguard"), [理财头脑](https://www.bogleheads.org/wiki/Rebalancing?utm_source=chatgpt.com "Rebalancing"))
* **CN10Y** ：中国外汇交易中心（CFETS/中国货币网）与中债收益率页面；程序内置多源回退与本地缓存，保证健壮性。([shibor.org](https://www.shibor.org/english/bmkierirt/?utm_source=chatgpt.com "Interest Rate (SDDS) - CFETS"), [ChinaBond Yield Curves](https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/more?locale=en_US&utm_source=chatgpt.com "CGB Yield Curve and Others - ChinaBond"))
* **定投/分批 vs 一次性** ：Vanguard 面向投资者的研究与解读。([investor.vanguard.com](https://investor.vanguard.com/investor-resources-education/news/lump-sum-investing-versus-cost-averaging-which-is-better?utm_source=chatgpt.com "Lump-sum investing versus cost averaging: Which is better?"), [先锋投资](https://corporate.vanguard.com/content/dam/corp/research/pdf/cost_averaging_invest_now_or_temporarily_hold_your_cash.pdf?utm_source=chatgpt.com "Cost averaging: Invest now or temporarily hold your cash?"))
* 开源库：`akshare`、`pandas`、`numpy`、`yfinance`、`pandas-datareader`、`requests`、`beautifulsoup4` 等。

> 以上外部资料仅用于理念阐释与数据口径参考；实际交易请以你账户的**实时行情与费税规则**为准。

---

## 注意与声明

* **不构成投资建议** 。本工具仅为**辅助决策**与 **执行流程管理** 。
* 开放式/LOF 的“估算净值”与当日收盘净值可能存在差异；下单以实际成交/估值为准。
* 请定期回看你的**风险承受能力**与 **目标权重** ，随人生阶段变化而调整。

## 许可证

MIT（见 `LICENSE`）。
