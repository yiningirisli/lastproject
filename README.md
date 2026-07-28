# lastproject

# Crypto Cross-Section — Day 1 Raw Data Layer

This repository implements the Day 1 sourcing stage for MMF1927H Option 4:
Crypto Cross-Section. It downloads raw daily Binance spot klines for the
candidate USDT universe and preserves the vendor response without cleaning or
feature engineering.

## Research scope

- **Source:** Binance public REST API (no account or API key required)
- **Endpoints:** `/api/v3/exchangeInfo` and `/api/v3/klines`
- **Frequency:** daily (`1d`)
- **Default sample:** 2018-01-01 through the current UTC date
- **Raw universe:** every currently trading USDT spot pair that passes the
  documented exclusions below
- **Downstream universe:** rank candidates using trailing, historically
  observable Binance quote volume at each rebalance date. Ranking is not
  performed during ingestion, because doing so using today's ranking would
  introduce look-ahead bias.

The 2018 start captures the 2018 crypto drawdown, the 2020–21 bull market, the
2022 rate-driven selloff, and subsequent regimes. Many assets listed later and
therefore have shorter histories.

## Candidate-universe rule

The raw layer excludes:

- stablecoin and fiat-like base assets;
- leveraged tokens (`UP`, `DOWN`, `3L`, `3S`, `5L`, `5S`);
- wrapped or duplicate exposures such as WBTC and WETH;
- symbols that are not currently trading or do not permit spot trading.

Pulling the full candidate set instead of today's top-N preserves the data
needed to form a trailing-volume top-N universe separately at every historical
rebalance date.

## Known limitation

`exchangeInfo` exposes only symbols known to Binance at pull time. Assets
delisted before the pull are absent, so the historical candidate set remains
survivorship-biased. The likely direction is upward: failed or delisted assets
are disproportionately omitted, which can overstate historical portfolio
performance. The manifest records this limitation explicitly. A truly
point-in-time listing history would be needed to eliminate it.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

All direct dependencies are pinned exactly for reproducibility.

## Usage

Small smoke pull:

```bash
python pull_binance_raw.py --start 2024-01-01 --limit-symbols 5
```

Full Day 1 pull:

```bash
python pull_binance_raw.py --start 2018-01-01
```

Specify an exclusive end date:

```bash
python pull_binance_raw.py --start 2024-01-01 --end 2024-02-01
```

Existing symbol files are skipped. Use `--force` only when an intentional
replacement is required.

## Raw output

Each run writes to:

```text
data/raw/binance_klines_1d_YYYY-MM-DD/
├── exchange_info.json
├── universe_candidates.json
├── provenance_log.jsonl
├── run_summary.json
└── SYMBOL.json
```

`exchange_info.json` is the untouched exchange metadata response.
Each symbol file contains the untouched 12-field kline arrays plus a provenance
wrapper. `run_summary.json` records requested dates, counts, failures, and
completion status.

The raw directory is intentionally ignored by Git because a full pull is large.
Commit the lightweight validation artifacts under `validation/`, while retaining
the immutable raw files in shared project storage.

## Validation

Run the offline tests:

```bash
python -m unittest discover -s tests -v
```

After a live pull, validate its structure:

```bash
python validate_raw.py data/raw/binance_klines_1d_YYYY-MM-DD
```

The validator exits nonzero for malformed symbol payloads, duplicate or
unordered timestamps, candles outside the requested half-open interval, or a
summary that reports failures.

## Day 1 / Day 2 boundary

Day 1 stores vendor responses and provenance only. Type conversion, missing-data
handling, outlier checks, panel alignment, historical top-N construction,
returns, features, and modeling belong downstream and must not mutate this raw
layer.

# 加密货币横截面项目 — Day 1 原始数据层

本项目实现 MMF1927H Workshop Option 4（Crypto Cross-Section）的 Day 1
数据获取阶段：从 Binance 下载 USDT 现货候选币的日频 K 线，并在任何清洗、
特征构造或建模之前，完整保存供应商返回的原始数据。

## 研究范围

- **数据源：** Binance 公共 REST API，无需账户或 API Key
- **API：** `/api/v3/exchangeInfo` 和 `/api/v3/klines`
- **频率：** 日频（`1d`）
- **默认样本期：** 2018-01-01 至运行当天的 UTC 日期
- **原始候选池：** 所有符合筛选规则、当前可交易的 USDT 现货交易对
- **后续投资池：** 在每个历史调仓日，利用当时可观察的历史 Binance
  quote volume 对候选币排名

数据获取阶段不会直接选择“今天成交量或市值最高的前 N 个币”。如果把今天的
排名应用到历史时期，会引入前视偏差。项目先获取完整候选池，再在 Day 2
使用历史滚动成交额构建每个调仓日的 Top-N 投资池。

默认从 2018 年开始，是为了覆盖不同的加密货币市场状态，包括：

- 2018 年加密货币熊市；
- 2020–2021 年牛市；
- 2022 年加息驱动的市场下跌；
- 此后的复苏及其他市场阶段。

部分币种上市较晚，因此其实际历史会短于完整样本期。

## 候选币筛选规则

原始候选池排除：

- 稳定币和类似法币的 base asset；
- 杠杆代币，例如 `UP`、`DOWN`、`3L`、`3S`、`5L` 和 `5S`；
- WBTC、WETH 等包装或重复风险敞口；
- 当前状态不是 `TRADING` 的币种；
- 不允许现货交易的币种；
- quote asset 不是 USDT 的交易对。

筛选规则、最终候选名单和每个被排除币种的原因都会写入
`universe_candidates.json`。

## 已知限制与偏差

Binance 的 `exchangeInfo` 主要反映抓取时仍可识别的交易对，无法完整恢复在
抓取日前已经退市的币种。因此，历史候选池仍然存在**幸存者偏差**。

该偏差可能使回测收益偏高，因为失败或退市的币种更容易从样本中消失。项目会
在 manifest 和 README 中明确披露这个限制。若要彻底消除该偏差，需要获得
真正的 point-in-time 历史上市与退市记录。

## 环境安装

建议使用 Python 3.11 或更高兼容版本。

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 固定了直接依赖和传递依赖的具体版本，以提高可复现性。

## 使用方法

### 小规模冒烟测试

先抓取少量币种，确认网络、API 和输出目录工作正常：

```bash
python pull_binance_raw.py --start 2024-01-01 --limit-symbols 5
```

### 完整 Day 1 抓取

```bash
python pull_binance_raw.py --start 2018-01-01
```

### 设置不包含在结果内的结束日期

```bash
python pull_binance_raw.py --start 2024-01-01 --end 2024-02-01
