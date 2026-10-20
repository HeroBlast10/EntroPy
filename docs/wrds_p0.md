# WRDS P0 correctness batch

The independent configuration is `config/wrds_research.yaml`. Its price,
factor, universe and output paths use `data/wrds/p0/`, with a 2025-12-31
cutoff. Earlier `data/wrds/repaired/` backtests and legacy Yahoo/ML results
are separate experiments. They do not inherit the new acceptance checks.

## Contracts

1. Default ML features come from an explicit built-in allowlist. New custom
   features require an explicit declaration. Identifiers and provenance are
   forbidden even in explicit lists. Training row ordering, bagging, capped
   sampling and prediction normalization are independent of PERMNO numbering.
   This also applies to legacy Ridge/Lasso/ElasticNet and their cross-validation
   and walk-forward wrappers; custom columns use `feature_cols=[...]` in the
   model constructor. MLP CUDA execution requires deterministic algorithms.
2. Returns dated t cover (t-1,t]. Direct CIZ total returns take precedence
   over adjusted-price ratios. CIZ delisting returns are already embedded.
   The shared calendar is the complete supplied market tape; never prefilter
   it to a security's observed rows or to today's index members. Ordinary
   gaps are unknown, including gaps between otherwise valid endpoints.
3. A multi-session label compounds through terminal settlement once, then
   holds zero-return cash through its original `label_end`. Missing terminal
   payoffs raise unless an explicit sensitivity policy is supplied. Labels
   extending past the tape remain missing. Training checks `label_end` before
   validation as well as applying the configured purge. Prediction rows do
   not depend on whether their future labels are available.
4. The overnight signal is an arithmetic mean of distribution-inclusive
   overnight returns. Define `(1+RET) = (1+overnight) * close/open` using raw
   same-day open/close. This cancels split units and attributes distributions
   to overnight. This convention is recorded explicitly rather than treating
   a missing adjustment factor as 1. Legacy data requires an actual adjustment
   factor. Nonpositive or untradable OHLC observations cannot form this signal.
5. Company financial numerators divide by the sum of the day's CRSP common
   equity class values for the historical PERMCO, including classes outside
   the index. Values convert from thousands of USD once. Company financials
   reach a secondary class only through an unambiguous, active dated CCM
   link at that date. Conflicts, unknown class values, expired links and
   unknown reporting currencies suppress the corresponding ratios. New
   issuer/class relationships are never backfilled. Security-level market
   cap is retained separately for trading/universe decisions.
   This is the CRSP-observable common-equity scope; unlisted/private classes
   have no independently observed CRSP market price and are not manufactured
   as extra daily values. An issuer-total valuation beyond this scope needs
   separately verified share-class disclosures and a stated pricing rule.
6. Membership/identity and financial vintage have independent admission
   gates. `--allow-restated-fundamentals` admits revised fundq research only.
   The deprecated `--allow-non-pit-research` has the same narrow scope; it
   cannot waive historical membership, stable PERMNO aliases or unique keys.
   Neither flag turns revised fundq into strictly PIT financial data.

## Rebuild and verify

Use the authorized WRDS account and the original membership, CCM and CIZ
downloads. First prepare the version-aware financial sources as described
in the [data dictionary](data_dictionary.md#wrds-research-data-contract).
Then run:

```powershell
python scripts/rebuild_wrds_p0.py --refresh-company
python scripts/verify_wrds_p0.py
python -m pytest -q
```

Subsequent rebuilds omit `--refresh-company` to reuse the saved company-class
tape. Raw licensed data stays in ignored `data/wrds/`; credentials and data
must not be added to Git. The 2026-10-20 company pull contains 2,959,206 daily
rows across 879 securities, including 22 outside the original 857-security
tape. There are 57,267 company dates with multiple common classes.

The acceptance script checks hashes against the prior raw inputs, reconciles
direct returns, validates membership/identity, recomputes company caps and
checks class-ratio equality. It independently loops through 231 real terminal
paths for the day-2/20-session scenario. Two paths use explicitly flagged
total-loss assumptions because their observed payoff is missing. It also
checks the overnight/intraday decomposition on approximately 2.9m valid rows.
Results are saved in `p0_acceptance.json`, `terminal_label_reconciliation.csv`
and `factor_coverage.csv` inside the P0 output directory.

Commit the WRDS source before model evaluation. The source scope, recorded
in run metadata, excludes concurrent paper-broker work. Tracked modifications
and untracked source files inside the WRDS scope block a frozen run. Each
artifact records its code commit, package versions and input hashes.

```powershell
# Full research run; total_loss is an explicit SIVB/SBNY sensitivity.
python scripts/evaluate_nonlinear_alpha.py --config config/wrds_research.yaml --allow-restated-fundamentals --missing-terminal-return total_loss
python scripts/evaluate_alpha_after_costs.py --config config/wrds_research.yaml --allow-restated-fundamentals

# Bounded integration smoke: validates interfaces, not model performance.
python scripts/evaluate_nonlinear_alpha.py --start 2018-01-01 --end 2020-12-31 --min-train-days 252 --validation-days 42 --n-splits 1 --lightgbm-trees 15 --lightgbm-max-train-rows 50000 --mlp-epochs 1 --mlp-max-train-rows 50000 --allow-restated-fundamentals --output data/wrds/p0/smoke_h1
python scripts/evaluate_alpha_after_costs.py --oof data/wrds/p0/smoke_h1/oof_predictions.parquet --output data/wrds/p0/smoke_h1/after_costs --allow-restated-fundamentals
```

For a 20-session smoke use `evaluate_nonlinear_alpha.py --horizon 20` with the
same bounded dates and a separate output directory. After-cost comparison
currently supports only 1-session OOF labels and rejects other horizons.
The short smoke and repaired factor coverage do not establish an improvement
in out-of-sample Sharpe. Full walk-forward retraining and cost comparisons
remain separate experiments.
