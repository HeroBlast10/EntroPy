# EntroPy — Data Dictionary

> Auto-generated reference for the three canonical Parquet tables produced by
> the data pipeline.  All tables are keyed by **(date, ticker)**.

---

## 1. `prices.parquet`

Daily OHLCV with point-in-time adjustment factor for US equities.

| Column        | Type      | Nullable | Description |
|---------------|-----------|----------|-------------|
| `date`        | `date32`  | No       | Trading date (NYSE calendar). |
| `ticker`      | `string`  | No       | Ticker symbol (e.g. `AAPL`). |
| `open`        | `float64` | Yes      | **Unadjusted** opening price. |
| `high`        | `float64` | Yes      | **Unadjusted** daily high. |
| `low`         | `float64` | Yes      | **Unadjusted** daily low. |
| `close`       | `float64` | Yes      | **Unadjusted** closing price. |
| `volume`      | `int64`   | Yes      | Share volume. |
| `amount`      | `float64` | Yes      | `close × volume` — turnover proxy. |
| `adj_factor`  | `float64` | Yes      | Cumulative split-adjustment factor **as-of `date`** (point-in-time). Multiply raw prices by this to get split-adjusted prices. |
| `adj_close`   | `float64` | Yes      | `close × adj_factor` — convenience column. |
| `is_tradable` | `bool`    | Yes      | `False` if the stock was halted or had zero volume. |

### Design Notes — Adjustment Factor

- We store **unadjusted** prices plus a separate `adj_factor` so users can
  choose between split-only and total-return adjustment downstream.
- When `adjustment.point_in_time = true` (default), `adj_factor` on date *t*
  reflects **only splits that had already occurred by *t***.  This prevents
  the common look-ahead bug where a future split retroactively changes
  historical prices.

---

## 2. `universe.parquet`

Daily tradable universe after applying listing, price, and market-cap filters.

| Column             | Type      | Nullable | Description |
|--------------------|-----------|----------|-------------|
| `date`             | `date32`  | No       | Trading date. |
| `ticker`           | `string`  | No       | Ticker symbol. |
| `days_since_ipo`   | `int32`   | Yes      | Calendar days since first appearance in `prices`. |
| `market_cap`       | `float64` | Yes      | Market capitalisation (USD) as-of `date`. |
| `close_price`      | `float64` | Yes      | Unadjusted close (same as in `prices`). |
| `in_index`         | `bool`    | Yes      | Member of target index (e.g. S&P 500) on `date`. |
| `pass_all_filters` | `bool`    | Yes      | `True` if the row passes **all** filters below. |

### Filters Applied (configurable in `settings.yaml`)

| Filter              | Config Key            | Default   | Logic |
|---------------------|-----------------------|-----------|-------|
| Listing age         | `universe.min_listing_days` | 60 days   | `days_since_ipo >= N` |
| Penny stock         | `universe.min_price`        | $1.00     | `close >= min_price` |
| Micro-cap           | `universe.min_market_cap`   | $50 M     | `market_cap >= threshold` |
| Halted / no trades  | *(from prices)*             | —         | `is_tradable == True` |

Only rows with `pass_all_filters == True` are persisted.

---

## 3. `fundamentals.parquet`

Point-in-time fundamental data (income statement, balance sheet, cash flow)
with publication lag applied to avoid look-ahead bias.

| Column                  | Type      | Nullable | Description |
|-------------------------|-----------|----------|-------------|
| `date`                  | `date32`  | No       | First **trading day** the data is considered "known" (`publish_date + lag`, snapped to calendar). |
| `ticker`                | `string`  | No       | Ticker symbol. |
| `report_date`           | `date32`  | Yes      | Fiscal period end date. |
| `publish_date`          | `date32`  | Yes      | SEC filing / press-release date. |
| `revenue`               | `float64` | Yes      | Total revenue. |
| `gross_profit`          | `float64` | Yes      | Gross profit. |
| `operating_income`      | `float64` | Yes      | Operating income (loss). |
| `net_income`            | `float64` | Yes      | Net income. |
| `eps_diluted`           | `float64` | Yes      | Diluted earnings per share. |
| `total_assets`          | `float64` | Yes      | Total assets. |
| `total_liabilities`     | `float64` | Yes      | Total liabilities. |
| `total_equity`          | `float64` | Yes      | Total stockholders' equity. |
| `cash_and_equivalents`  | `float64` | Yes      | Cash, cash equivalents & short-term investments. |
| `total_debt`            | `float64` | Yes      | Total debt (short + long term). |
| `cash_from_operations`  | `float64` | Yes      | Net cash from operating activities. |
| `capex`                 | `float64` | Yes      | Capital expenditures (negative = outflow). |
| `free_cash_flow`        | `float64` | Yes      | `cash_from_operations + capex`. |
| `market_cap`            | `float64` | Yes      | Market capitalisation (USD). |
| `shares_outstanding`    | `float64` | Yes      | Diluted shares outstanding. |
| `book_value_per_share`  | `float64` | Yes      | `total_equity / shares_outstanding`. |

### Point-in-Time (PIT) Handling

```
available_date = publish_date + publication_lag_days   (default 45 cal days)
date           = next_trading_day(available_date)
```

This ensures that on any given backtest date *t*, only information that was
**publicly available before *t*** is used — eliminating look-ahead bias
from late filings, restatements, or data vendor back-fill.

---

## 4. `manifest.json`

Metadata file recording the exact provenance of each data build.

| Field              | Type     | Description |
|--------------------|----------|-------------|
| `build_timestamp`  | ISO 8601 | UTC time the build completed. |
| `config_hash`      | string   | xxHash-64 of `settings.yaml`. |
| `git_head`         | string   | Short SHA of `HEAD` at build time. |
| `file_checksums`   | object   | `{filename: xxHash-64}` for every Parquet file. |
| `row_counts`       | object   | `{table_name: int}` for quick sanity checks. |
| `date_range`       | object   | `{start, end}` from config. |
| `git_tag`          | string   | Auto-created tag (e.g. `data-v20240601-153000`). |

Use `python scripts/build_dataset.py --verify` to re-hash all files and
compare against the manifest.

---

## Conventions

- **Date type**: All `date` columns are stored as `date32` (calendar date, no
  time component).
- **Currency**: All monetary values are in **USD**.
- **NaN semantics**: A `NaN` means the data point is unavailable for that
  (date, ticker) pair — not that the value is zero.
- **Compression**: Parquet files use **zstd** compression.
- **Sort order**: Every table is sorted by `(date, ticker)`.
