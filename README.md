# 资产再平衡助手

这是一个用于辅助投资组合再平衡的 Python 工具，帮助投资者根据设定的目标权重，自动计算并生成分批加/减仓计划。

## 功能特点

- **平滑到位**：将与目标的差额按设定周数分摊，再乘以强度系数 (0/0.5/1)
- **双方案输出**：
  - 方案A（交易约束）：ETF 按100股整手、减持仅动开放式/LOF
  - 方案B（纯基金）：全部按基金处理，任意金额买卖（便于按权重铺开）
- **智能强度调节**：根据估值分位 + 回撤加权（贵时减速，跌时稍加速）
- **再平衡带宽**：未明显偏离目标不动，降低高位小额补仓的概率
- **自动数据获取**：通过多个数据源获取最新行情和估值数据
- **配置文件分离**：支持默认配置和自定义配置，便于开源和个人使用

## 安装依赖

```bash
pip install akshare pandas numpy yfinance pandas-datareader
```

## 配置说明

1. **目标权重配置**：
   在 `rebalance_helper.py` 中修改 `targets` 字典，设置各资产的目标权重

2. **持仓配置**：
   - 默认配置：`config/default_positions.py`（示例数据，用于开源）
   - 自定义配置：`config/custom_positions.py`（实际使用的数据，不会被 git 跟踪）

3. **全局参数**：
   在 `rebalance_helper.py` 中可调整以下参数：
   - `TOTAL_TARGET`：目标总额
   - `STAGING_WEEKS`：计划用多少周逐步到位
   - `WEEKLY_LIMIT`：每周最大操作金额上限
   - `USE_INTENSITY`：是否启用强度系数
   - `BAND_PP`：再平衡带宽（百分点）
   - `BOARD_LOT_CODES`：需要按100股整手成交的代码
   - `SELLABLE_CODES`：可任意金额减持的代码

## 使用方法

### 基本使用

```bash
python rebalance_helper.py
```

### 命令行参数

```bash
# 仅生成方案A（交易约束）
python rebalance_helper.py --plan A

# 仅生成方案B（纯基金）
python rebalance_helper.py --plan B

# 不导出CSV文件
python rebalance_helper.py --no-export

# 指定CSV导出目录
python rebalance_helper.py --outdir my_exports
```

## 输出内容

1. **控制台输出**：
   - 关键指标（PE、收益率缺口、估值分位等）
   - 资金进度和分批参数
   - 买入/减持清单摘要
   - 持仓配比表

2. **CSV导出**（默认到 `exports` 目录）：
   - 买入订单清单（`buy_orders_*.csv`）
   - 减持订单清单（`sell_orders_*.csv`）
   - 持仓前后对比（`holdings_before_*.csv`、`holdings_after_*.csv`、`holdings_diff_*.csv`）
   - 目标权重表（`targets_*.csv`）

## 自定义配置

首次使用时，请创建自己的持仓配置文件：

1. 复制 `config/default_positions.py` 到 `config/custom_positions.py`
2. 在 `custom_positions.py` 中修改 `current_positions` 字典，填入实际持仓数据

## 注意事项

- 开放式/LOF 使用的是"估算净值"，与收盘净值存在偏差
- 下单以当时成交价/估值为准
- 导出的 CSV 建议在表格软件里查看与执行

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
