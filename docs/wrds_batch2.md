# WRDS second batch: complete factors and fair model comparison

This batch migrates the original 42 factors and adds two separately named true
share-turnover factors. It does not rename relative-volume proxies into share
turnover. Historical SIC/NAICS data are independently pulled from the verified
`crsp.stksecurityinfohist` catalog: 4,453 intervals covering 857 securities.
The original raw downloads and the first batch remain separate inputs.

## Economic and temporal contracts

- Raw factors use the full shared trading calendar, stable PERMNO aliases and
  CIZ direct total returns. Cross-sectional ranking happens on lagged historical
  eligibility, after signal lags. Financial factors reuse the version-aware P0
  adapter, including company-level cap and dated CCM attribution. The registry
  momentum convention is P[t-21]/P[t-252]-1; the older P0 baseline has an extra
  decision lag and is not a same-definition performance comparison.
- Kalman state starts at the first observed price, can recover from missing
  observations, and does not manufacture a signal on a missing quote. Spectral,
  Hurst and higher-moment windows use direct returns and reject incomplete
  windows. The spectral DFT basis is cached without changing its periodogram.
- OU is a signed single-stock mean-reversion signal, not a pair strategy.
  Positive price deviation has negative alpha direction; unavailable stationary
  fits stay missing. Path momentum uses a drawdown contained in its formation
  window. Two-day high/low ranges use comparable adjusted price units.
- Volume cache and uncached paths share chronological alignment and minimum
  observations. Market/size proxies use lagged index eligibility and lagged
  size; stocks joining the index later do not retrospectively enter proxies.
- SIC history is joined only inside effective intervals and lagged one session
  for decisions. Two-digit SIC major groups provide industry controls. SIC
  60–67 identify finance/insurance/real estate per the [official SIC manual](https://www.osha.gov/data/sic-manual/division-h).
  Effective-date history is not an as-reported vintage product: classification
  provenance remains explicitly non-strict-PIT, as do restated fundq features.
- Financial raw versus industry/size residual variants use matched available
  rows. The residual uses a joint projection on industry and log size; sequential
  projections that reintroduce industry exposure are avoided. Financial and
  real-estate exclusions are predeclared sensitivity groups, not chosen after
  seeing results.

## Diagnostics

`diagnose_wrds_factors.py` reports 1/5/20/60-session Rank IC, calendar-aware HAC,
exploratory BH correction, annual stability, industry/size/past-volatility
groups, long/short gross horizon spreads, signal turnover, equal-weight-universe
beta, industry exposures, residual IC versus momentum/profitability and rank
redundancy. Raw gross overlapping horizon spreads are not executed returns.
The raw scorecard is descriptive; none of its full-history rankings chooses
features or signs for the model experiment.

## Model protocol

`config/wrds_batch2.yaml` declares all 44 possible predictors, two information
blocks, two Ridge configurations and two conservative LightGBM configurations.
Feature coverage/variance gates are locked before the first INNER validation
period. Both information blocks predict identical securities/dates, retain
unlabeled inference rows and use identical capped training samples. Missing
feature ranks use a fixed zero midpoint; no full-sample imputer is fitted.

The primary task is 20 sessions. Each outer expanding fold contains a purged
inner validation window; parameter choice never accesses outer labels. Three
fixed seeds are averaged for LightGBM. Five-session robustness reuses the
primary-horizon parameter choices without another search. The declared budget
is 12 model/information/horizon configurations; the full trial registry also
records inner/outer fits, seeds, training-key hashes and a fixed unselected
shuffled-prediction placebo. Neither configuration count nor a paired interval
is represented as a global multiple-testing-adjusted DSR.

Equal-weight combinations and fixed momentum/profitability baselines use the
same OOF prediction universe. Rebalance every 20/5 sessions respectively,
with gross target at most 1, 1.5% stock and 3% issuer gross target caps, $1m
capital and 10% ADV execution limits. Twofold cost stress is a full execution
replay. Missing SIVB/SBNY payoffs require explicit total-loss/zero-return runs;
the sensitivity uses frozen primary choices. Neither is an observed payoff.
The tape ends 2025-12-31 and the last marked period may be a shorter stub.

2023–2025 are already examined historical data. Outer folds are a historical
review, not an untouched final test. Paired IC/net-return intervals, stress
results, placebo and all attempted settings remain visible. Revised fundamental
inputs prevent production admission regardless of historical performance.
Order lifecycle, independent position/cash attribution and capital-scale full
replays remain subsequent work; residual non-rebalance costs are measured here.

## Commands

Save the authorized historical industry pull under `data/wrds/batch2/` using
`scripts.wrds_ingest.industry.pull_industry_history`. The original P0 inputs
must pass their first-batch audit. Licensed data and credentials remain ignored.

```powershell
python scripts/build_wrds_factor_library.py --jobs 4
# Freeze the source. Cached parts can then be reassembled after their source
# and prepared-input hashes are verified; changed implementations recompute.
python scripts/build_wrds_factor_library.py --reuse-parts --jobs 4
python scripts/diagnose_wrds_factors.py --missing-terminal-return total_loss
python scripts/compare_wrds_models.py --allow-restated-fundamentals --missing-terminal-return total_loss --smoke
python scripts/compare_wrds_models.py --allow-restated-fundamentals --missing-terminal-return total_loss
python scripts/compare_wrds_models.py --allow-restated-fundamentals --missing-terminal-return zero_return --sensitivity-from data/wrds/batch2/model_comparison/total_loss/full
python scripts/verify_wrds_batch2.py --require-models
python -m pytest -q
```

Main outputs: `factor_catalog.csv`, `factors.parquet`, `diagnostics/`, nested
`trial_registry.json`, `locked_feature_schema.csv`, `fold_audit.csv`, complete
OOF predictions, IC and paired net-return comparisons, baseline/stress execution
records and `batch2_acceptance.json`. Earlier results are not overwritten.
