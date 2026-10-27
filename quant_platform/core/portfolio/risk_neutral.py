"""Convex factor-risk portfolios with explicit neutrality and turnover costs.

Uses the fitted FactorRiskModel's annual covariance and specific volatility.
The objective converts them to the same daily units as the alpha prior. The
factor representation avoids forming a dense N-by-N covariance every day.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from quant_platform.core.portfolio.optimize import PortfolioOptimizationError


@dataclass(frozen=True)
class NeutralRiskConfig:
    gross_limit: float = 1.0
    stock_limit: float = .015
    issuer_limit: float = .03
    expected_alpha_annual: float = .10
    risk_aversion: float = 3.0
    turnover_cost_bps: float = 10.0
    forecast_horizon: int = 20
    target_volatility: float = .10
    neutral: bool = True

    def __post_init__(self):
        for name in ("gross_limit", "stock_limit", "issuer_limit", "expected_alpha_annual", "target_volatility"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.risk_aversion < 0 or self.turnover_cost_bps < 0 or self.forecast_horizon < 1:
            raise ValueError("Invalid objective coefficients")


def industry_quantile_weights(day, score, *, stock_limit=.015, issuer_limit=.03, min_group=5):
    """Equal-weight tails within each industry; unmatched sleeves stay in cash."""
    result = pd.Series(0., index=day.index)
    known = day[score].notna() & day.decision_industry.notna()
    total = int(known.sum())
    if not total:
        return result
    for _, group in day.loc[known].groupby("decision_industry", observed=True):
        if len(group) < min_group:
            continue
        rank = group[score].rank(pct=True)
        long, short = rank.ge(.8), rank.le(.2)
        if not long.any() or not short.any():
            continue
        budget = .5 * len(group) / total
        result.loc[group.index] = budget * (long.astype(float) / long.sum() - short.astype(float) / short.sum())
    result = result.clip(-stock_limit, stock_limit)
    issuer = day.permco.fillna(-day.permno)
    gross = result.abs().groupby(issuer).transform("sum")
    result *= np.minimum(1, issuer_limit / gross.replace(0, np.nan)).fillna(1)
    # Clipping can imbalance an industry's two sides. Shrink the larger side,
    # never enlarge a capped name or silently reintroduce sector exposure.
    for _, group in day.loc[known].groupby("decision_industry", observed=True):
        w = result.loc[group.index]
        pos, neg = w.clip(lower=0).sum(), -w.clip(upper=0).sum()
        budget = min(pos, neg)
        result.loc[w.index[w.gt(0)]] *= budget / pos if pos else 0
        result.loc[w.index[w.lt(0)]] *= budget / neg if neg else 0
    return result


def solve_factor_neutral(scores, model, industries, issuers, market_beta, previous,
                         config=NeutralRiskConfig(), warm_state=None):
    """Dollar neutral always; optionally beta and each industry net=0.

    Turnover is penalized against ACTUAL marked holdings. A uniform 10 bps
    L1 proxy is an optimization assumption; realized fills still pay the full
    original commission/slippage/impact/borrow model in the execution ledger.
    Zero holdings are feasible, so no forced leverage or silent fallback is
    required. Volatility scaling only reduces risk under the gross cap.
    """
    import osqp

    names = scores.index
    if len(names) < 3 or scores.isna().any():
        raise PortfolioOptimizationError("Too few finite scores for factor optimization")
    b = model.exposures_.reindex(names).to_numpy(dtype=float)
    specific = model.specific_risk_.reindex(names).to_numpy(dtype=float)
    f = np.asarray(model.cov_matrix_, dtype=float)
    beta = market_beta.reindex(names).to_numpy(dtype=float)
    industry = industries.reindex(names)
    issuer = issuers.reindex(names)
    if not all(np.isfinite(x).all() for x in (b, specific, f, beta)) or industry.isna().any() or issuer.isna().any():
        raise PortfolioOptimizationError("Incomplete risk/neutrality context")
    if (specific < 0).any() or not np.allclose(f, f.T, atol=1e-10):
        raise PortfolioOptimizationError("Invalid factor covariance or specific risk")
    n, k = b.shape
    # Variables: holdings w, |w|, |w-w_actual|, portfolio factor exposures g.
    width = 3 * n + k
    rank = scores.rank(method="average").to_numpy()
    alpha = (rank - (n + 1) / 2) / max(n - 1, 1) * config.expected_alpha_annual / 252
    prior = previous.reindex(names, fill_value=0).to_numpy(dtype=float)
    h = config.forecast_horizon
    p = sparse.block_diag((sparse.diags(config.risk_aversion * h * specific ** 2 / 252 + 1e-8),
                           sparse.csc_matrix((2 * n, 2 * n)), sparse.csc_matrix(config.risk_aversion * h * f / 252)), format="csc")
    q = np.r_[-h * alpha, np.zeros(n), np.full(n, config.turnover_cost_bps / 10000), np.zeros(k)]
    identity, zero = sparse.eye(n, format="csc"), sparse.csc_matrix((n, n))
    factor_zero = sparse.csc_matrix((n, k))
    blocks, lower, upper = [], [], []

    def add(matrix, low, high):
        blocks.append(sparse.csc_matrix(matrix))
        size = matrix.shape[0]
        lower.extend(np.broadcast_to(low, size)); upper.extend(np.broadcast_to(high, size))

    # Auxiliary absolute values and stock caps.
    add(sparse.hstack([-identity, identity, zero, factor_zero]), 0, np.inf)
    add(sparse.hstack([identity, identity, zero, factor_zero]), 0, np.inf)
    add(sparse.hstack([zero, identity, zero, factor_zero]), 0, config.stock_limit)
    add(sparse.hstack([-identity, zero, identity, factor_zero]), -prior, np.inf)
    add(sparse.hstack([identity, zero, identity, factor_zero]), prior, np.inf)
    add(sparse.hstack([zero, zero, identity, factor_zero]), 0, np.inf)
    add(sparse.csc_matrix(np.r_[np.zeros(n), np.ones(n), np.zeros(n + k)][None, :]), 0, config.gross_limit)
    issuer_matrix = pd.get_dummies(issuer, dtype=float).to_numpy().T
    add(sparse.hstack([sparse.csc_matrix((len(issuer_matrix), n)), sparse.csc_matrix(issuer_matrix),
                       sparse.csc_matrix((len(issuer_matrix), n + k))]), 0, config.issuer_limit)
    add(sparse.hstack([sparse.csc_matrix(b.T), sparse.csc_matrix((k, 2 * n)), -sparse.eye(k)]), 0, 0)
    controls = [np.ones(n)]
    if config.neutral:
        # Drop one industry dummy because global net=0 implies its balance.
        controls.extend(pd.get_dummies(industry, dtype=float).to_numpy().T[:-1])
        controls.append(beta)
    controls = np.asarray(controls)
    add(sparse.hstack([sparse.csc_matrix(controls), sparse.csc_matrix((len(controls), 2 * n + k))]), 0, 0)
    p = sparse.triu(p, format="csc")
    a = sparse.vstack(blocks, format="csc")
    p.sort_indices(); a.sort_indices()
    signature = (tuple(names), tuple(industry), tuple(issuer))
    reusable = (warm_state is not None and warm_state.get("signature") == signature
        and np.array_equal(warm_state["p_indptr"], p.indptr) and np.array_equal(warm_state["p_indices"], p.indices)
        and np.array_equal(warm_state["a_indptr"], a.indptr) and np.array_equal(warm_state["a_indices"], a.indices))
    if reusable:
        solver = warm_state["solver"]
        solver.update(Px=p.data, Ax=a.data, q=q, l=np.asarray(lower), u=np.asarray(upper))
        solver.update_settings(max_iter=30000)
    else:
        solver = osqp.OSQP()
        solver.setup(P=p, q=q, A=a, l=np.asarray(lower), u=np.asarray(upper), verbose=False,
                     eps_abs=1e-8, eps_rel=1e-8, max_iter=30000, polishing=True)
        if warm_state is not None:
            warm_state.update(solver=solver, signature=signature, p_indptr=p.indptr.copy(), p_indices=p.indices.copy(),
                              a_indptr=a.indptr.copy(), a_indices=a.indices.copy())
    # Actual marked holdings are also the center of the L1 penalty. Keep the
    # previous solve's dual state on reusable patterns, rather than starting
    # thousands of near-identical daily problems from zero.
    solver.warm_start(x=np.r_[prior, np.abs(prior), np.zeros(n), b.T @ prior])
    solution = solver.solve(raise_error=False)
    if solution.info.status_val != 1:
        solver.update_settings(max_iter=100000)
        solution = solver.solve(raise_error=False)
    if solution.info.status_val != 1:
        raise PortfolioOptimizationError(f"OSQP failed: {solution.info.status}")
    weights = pd.Series(solution.x[:n], index=names)
    # Auxiliary |w| constraints have finite solver accuracy; tiny errors can
    # accumulate across hundreds of names. A single downward scale enforces
    # every gross/stock/issuer cap while preserving all homogeneous neutrality
    # constraints. No relative weights or constraints are relaxed.
    gross = float(weights.abs().sum())
    stock = float(weights.abs().max())
    issuer_gross = float(weights.abs().groupby(issuer).sum().max())
    cap_scale = min(1., config.gross_limit / gross if gross else 1.,
                    config.stock_limit / stock if stock else 1.,
                    config.issuer_limit / issuer_gross if issuer_gross else 1.)
    weights *= cap_scale
    risk = model.decompose_risk(weights)
    pre_vol = risk["total_risk"]
    scale = min(1., config.target_volatility / pre_vol) if pre_vol > 0 else 1.
    weights *= scale
    exposures = weights.groupby(industry).sum()
    if (weights.abs().max() > config.stock_limit + 2e-6 or weights.abs().sum() > config.gross_limit + 2e-5
            or abs(weights.sum()) > 2e-6
            or weights.abs().groupby(issuer).sum().max() > config.issuer_limit + 2e-6):
        raise PortfolioOptimizationError(f"QP output violates position constraints: gross={weights.abs().sum()}, "
            f"stock={weights.abs().max()}, issuer={weights.abs().groupby(issuer).sum().max()}, net={weights.sum()}")
    if config.neutral and (exposures.abs().max() > 2e-6 or abs(weights @ market_beta.reindex(names)) > 2e-6):
        raise PortfolioOptimizationError("QP output violates beta/industry neutrality")
    return weights, {"predicted_volatility": pre_vol * scale, "pre_scale_volatility": pre_vol,
                      "volatility_scale": scale, "volatility_target_reached": pre_vol >= config.target_volatility,
                      "feasibility_scale": cap_scale,
                      "target_market_beta": float(weights @ market_beta.reindex(names)),
                      "target_industry_abs_net": float(exposures.abs().sum()),
                      "solver_iterations": solution.info.iter, "solver_status": str(solution.info.status)}
