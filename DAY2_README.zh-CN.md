# Day 2 — 数据工程与清洗

`day2_clean.py` 将 Day 1 保存的 Binance 原始 JSON 转换为一个经过验证、
可以直接供 Day 3 使用的 Parquet 面板。程序不会修改或覆盖原始数据层。

## 输出文件

```text
data/clean/
├── crypto_analysis_ready.parquet
├── day2_lineage_manifest.json
└── DAY2_DATA_QUALITY_MEMO.md
```

- `crypto_analysis_ready.parquet`：分析就绪的单一面板，使用 Zstandard 压缩；
- `day2_lineage_manifest.json`：记录全部参数、原始文件 SHA-256、清洗脚本
  hash、输出文件 hash、Git commit 和质量诊断；
- `DAY2_DATA_QUALITY_MEMO.md`：记录缺失值、异常值、投资池和剩余偏差。

## 运行方法

```bash
python day2_clean.py data/raw/binance_klines_1d_YYYY-MM-DD
```

也可以明确指定参数：

```bash
python day2_clean.py RAW_DIRECTORY \
  --out data/clean \
  --top-n 50 \
  --volume-window 30 \
  --min-history 30 \
  --winsor-lower 0.01 \
  --winsor-upper 0.99
```

## 清洗规则

### Schema 和数据有效性

每个币种文件必须包含 Day 1 provenance wrapper，以及 Binance 的 12 个 K
线字段。程序会：

- 显式转换数值字段；
- 按币种和日期排序；
- 删除重复的 `symbol-date`；
- 拒绝缺失或无法转换的 OHLCV；
- 拒绝非正价格和负成交量；
- 检查 `high`、`low`、`open`、`close` 的逻辑关系。

### 缺失值

收益率不会被填补。币种上市首日、缺少上一期价格或交易中断造成的缺失，与上市
及交易状态有关，不能安全地假设为 MCAR。

这些收益率保留为 `NaN`，同时设置 `return_missing=True`。程序不进行向后
插值，也不会利用未来观察值填补过去。

### Point-in-time 动态投资池

默认投资池是每个日期符合条件的成交额前 50 个币：

1. 使用 USDT quote volume；
2. 先将成交额滞后一天；
3. 计算过去 30 个观察日的平均成交额；
4. 在每个日期进行横截面排名；
5. 只选择排名前 50 的币；
6. 币种必须拥有超过 30 个观察日才可进入投资池。

因此，日期 t 的投资池只使用截至 t-1 已知的数据，不使用当天结束后才完整获得
的信息，也不会为新上市币种回填上市前历史。

### 异常值

合法的极端收益不会被删除。程序保留原始 `return_1d`，并另行生成：

- `return_1d_winsor`：在每个日期横截面内进行 p1/p99 winsorization；
- `return_1d_robust_z`：基于 median 和 MAD 的稳健 z-score。

这样既保留真实尾部信息，也为 Day 3 提供稳健版本。

### Day 3 预测目标

`target_return_1d` 是同一币种的下一期日收益。目标列与输入数据明确分开，避免
Day 3 构造特征时意外使用未来信息。

## 面板主要字段

- 标识：`date`、`symbol`；
- 原始行情：OHLC、成交量、quote volume、成交笔数和主动买入量；
- 收益率：简单收益率和对数收益率；
- 清洗字段：缺失标记、winsorized return、robust z-score；
- 投资池字段：历史长度、滞后滚动成交额、成交额排名、eligibility 和
  `in_universe`；
- 预测目标：下一期日收益。

## 已解决和未解决的问题

所有滚动计算都遵守 point-in-time 原则：未来价格和成交量不会进入更早日期。
程序允许不同币种拥有不同长度的历史，不会回填上市前数据。

但是 Day 1 的 `exchangeInfo` 快照无法恢复抓取日前已经退市的币种，因此仍有
剩余的幸存者偏差，并可能使历史策略表现偏高。代码会把这个限制写入 lineage
manifest 和 data-quality memo，而不会错误声称已经完全消除。

加密货币现货不存在股票拆分、分红、GICS 行业分类和期货换月，因此讲义中的
这些检查不适用于本项目。

## 测试

```bash
python -m unittest discover -s tests -v
```

Day 2 测试覆盖：

- schema 和数值类型转换；
- 重复值删除；
- 非法 OHLC 数据拒绝；
- 滞后、无前视的动态投资池；
- 收益率缺失不进行填补。
