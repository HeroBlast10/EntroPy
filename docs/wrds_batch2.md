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
Shared training sampling orders on date, common price features and training
target, using the same predictor union only to resolve exact ties. Identifiers
and provenance are excluded from both levels. Remaining ties have identical
predictors and targets in every information block. Relabeling security IDs
therefore cannot change the fitted data, even with capped sampling.

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
replay using the declared `CostModel` defaults, including doubled borrowing
costs. These are simulated cost assumptions, not observed broker fills or a
current broker tariff schedule. Missing SIVB/SBNY payoffs require explicit total-loss/zero-return runs;
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

## Measured diagnostic results (2026-10-21)

The rebuilt panel contains 2,903,815 observed security/date rows and 2,021,564
eligible research rows. All 44 registered factors have usable histories;
effective-dated industry coverage is 100% among eligible rows. The 224-row
scorecard covers the four declared horizons and matched financial ablations.
Financial coverage in the 2010–2025 eligible panel is:

| Factor | Coverage |
| --- | ---: |
| ASSET_GROWTH | 97.56% |
| GROSS_PROFITABILITY | 98.67% |
| ACCRUALS | 98.48% |
| BOOK_TO_MARKET | 99.72% |
| EARNINGS_YIELD | 99.01% |
| PIOTROSKI_F8 | 83.98% |

These coverage figures do not upgrade revised fundq into a strict-PIT product.
At 20 sessions no factor/variant passes the exploratory BH 5% threshold across
all 224 comparisons. Ten comparisons pass at one session; this does not establish
profitability after costs at that frequency. Matched industry/size adjustment
raises ACCRUALS' 20-session IC from 0.00347 to 0.00925 (HAC t=2.74), but its
multiple-testing-adjusted result still does not pass the declared threshold.

Redundancy diagnostics identify essentially identical raw daily ranks for the
two skewness implementations, the two kurtosis implementations and MOM_1M versus
STR_1M. The last pair has opposite declared alpha directions. High redundancy
also appears for MOM_6M/MOM_PATH and DOWNVOL_60D/TAIL_RISK. These are candidates
for a separately frozen, training-only ablation; they are retained in the
predeclared model comparison instead of removed after viewing OOF outcomes.

The model schema locks on 2021-04-30. Four outer folds run from 2021-12-27 through
2025-12-31, each with 252 prediction sessions. Each information block has 506,738
OOF security/date predictions per horizon, using 38 price/volume predictors or
44 predictors including six financial features. The 20-session LightGBM minus
Ridge IC delta with financials is 0.00056, with a paired 95% interval of
[-0.01529, 0.01666]. Financials minus price-only has an IC delta of -0.00284 for
Ridge and -0.00108 for LightGBM; both paired intervals cross zero. These results
do not support a demonstrated nonlinear or financial-information increment.

Default pytest discovery is restricted to `tests/`. An ignored experiment
snapshot contains 47 duplicate Tier A tests and previously inflated whole-tree
counts; it is no longer collected. The committed test suite contains 449
distinct test cases, including 31 new second-batch cases. Licensed row-level
inputs and generated model artifacts remain under ignored `data/wrds/`.

## Executed primary results

The total-loss research scenario for the two missing terminal payoffs produces
these results over 2021-12-27–2025-12-31 (1,008 marked sessions). This is a
historical OOF execution comparison, with $1m starting capital and gross target
at most one. Annualized returns below include simulated transaction and borrow
costs; the doubled-cost column comes from a separate execution replay.

| Horizon | Information | Model | Net annual return, base | Net annual return, 2x cost | Base net Sharpe |
| --- | --- | --- | ---: | ---: | ---: |
| 20 sessions | Price/volume | ridge | 0.873% | -0.680% | 0.2177 |
| 20 sessions | Price/volume | lightgbm | 0.777% | -0.700% | 0.2038 |
| 20 sessions | Price/volume + financial | ridge | -0.155% | -1.628% | -0.0091 |
| 20 sessions | Price/volume + financial | lightgbm | -0.892% | -2.168% | -0.1611 |
| 5 sessions | Price/volume | ridge | 0.003% | -4.529% | 0.0298 |
| 5 sessions | Price/volume | lightgbm | -1.438% | -5.803% | -0.2333 |
| 5 sessions | Price/volume + financial | ridge | -0.428% | -4.643% | -0.0480 |
| 5 sessions | Price/volume + financial | lightgbm | -2.373% | -6.286% | -0.4121 |

Fixed 20-session momentum has base net annual return 0.087% and net Sharpe
0.0540; fixed gross profitability has -1.889% and -0.2939 respectively.
The fixed shuffled Ridge controls lose money at both horizons and costs.
No tested Ridge/LightGBM information block remains profitable at doubled costs.
Thus the repairs improve data coverage, definitions and auditability without
establishing a robust profitable strategy. The earlier legacy net-Sharpe 1.09
headline must not be relabeled as a result of this WRDS experiment.

All eight primary paired net-return intervals include zero. In the financial
20-session comparison, LightGBM minus Ridge has mean daily net-return delta
-0.292 basis points, with a paired 95% interval [-1.731, 1.023] basis points.
These intervals describe the declared historical experiment, not a future
performance guarantee.

A final synthetic audit exposed an identifier dependence when price/target
sampling keys tie but financial predictors differ. The added secondary keys
and regression test fix that edge case. On the real 2,021,564-row panel there
are **zero** duplicate date/common-price-feature keys, even without target in
the key. Thus the recorded OOF selections and model results (frozen at
`35ddc5f`) are unchanged by this subsequent boundary fix; their original source
metadata are preserved. `sampling_tie_audit.json` records this check locally.

## Final acceptance and remaining work

Both terminal policies complete all 36 predeclared replays, for **72** formal
books in total, in addition to 36 smoke replays. Parameter choices for both
zero-return horizons exactly match the frozen primary 20-session choices.
The largest absolute change in annual net return across corresponding books
is 0.335 percentage points. Every Ridge/LightGBM block remains negative
at doubled costs under both policies. These scenarios are assumptions for the
two missing payoffs, not observations or proven bounds on the true outcome.

The final verifier passes factor/part/input hashes, effective industry dates,
seven actual-CRSP future-perturbation/prefix checks, purged label boundaries,
identical training and prediction keys, transaction-cost aggregation and the
daily net-NAV identity. The maximum executed ADV participation across both
policies is 0.0010300153 (0.1030%), below the 10% execution limit. Separate
checks on all books verify tradable-only integer-share fills, gross target at
most 1, maximum stock target 0.5% and issuer gross target 1.0%, below the
configured 1.5% and 3% caps. These are target constraints; realized exposure
may drift between rebalances.

`batch2_acceptance.json`, `terminal_policy_sensitivity.csv`, the paired
IC/net-return CSVs, `sampling_tie_audit.json` and `batch2_completion.json`
retain the full local evidence. Both policies have
`research_increment_supported=false` and `production_admitted=false`.

The next research work should first address independent positions/cash
reconciliation, residual-order lifecycle and cost/capital replay sensitivity.
A subsequent, separately frozen model experiment can test training-only
redundancy pruning and industry/size-adjusted financial features. Full-history
factor rankings must not be used to retune this already reviewed OOF period.
More parameter search, extra models or IBES features are not supported as the
immediate next action by this batch's evidence. A genuinely new untouched test
period and stronger fundamental vintages remain necessary for stronger claims.
