"""Close-to-close execution simulation with a stateful holdings ledger.

The convention used throughout this module is:

1. positions held after yesterday's close earn today's close-to-close return;
2. borrow is charged on the marked short book;
3. today's target is traded at today's close;
4. trading costs are deducted after the fills.

Consequently a target observed on date ``t`` never earns date ``t``'s return.
The simulator also keeps actual holdings separate from targets, so holdings
drift, missing prices, share rounding and participation limits are reflected
in both turnover and subsequent PnL.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.execution.cost_models.us_equity import (
    CostModel,
    daily_borrow_cost,
    estimate_batch_costs,
)


_TRADE_COLUMNS = [
    "ticker",
    "side",
    "weight_from",
    "weight_to",
    "delta_weight",
    "notional_trade",
    "requested_shares",
    "shares",
    "price",
    "adv_shares",
    "daily_vol",
    "executed_notional",
    "executed_delta_weight",
    "fill_ratio",
    "fill_status",
    "requested_participation",
    "executed_participation",
    "suppressed_notional",
]

_COST_COLUMNS = [
    "notional",
    "commission",
    "slippage",
    "impact",
    "sec_fee",
    "finra_taf",
    "stamp_duty",
    "total_cost",
    "total_cost_bps",
]

_INCOMPLETE_FILL_STATUSES = {
    "partial",
    "rejected_missing_price",
    "rejected_no_liquidity",
    "rejected_rounding",
    "rejected_not_tradable",
}


def generate_trades(
    weights_today: pd.Series,
    weights_yesterday: Optional[pd.Series],
    prices_today: pd.Series,
    adv_today: pd.Series,
    vol_today: pd.Series,
    capital: float,
    max_participation_rate: Optional[float] = None,
    tradable_today: Optional[pd.Series] = None,
    min_adv_shares: float = 0.0,
    target_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Convert target-minus-actual weights into executable orders.

    ``weights_yesterday`` means the *actual pre-trade* weights, rather than
    yesterday's stale target.  Rejected and partial orders remain in the
    returned table with an explicit ``fill_status``; callers must update
    holdings using ``executed_notional`` only.
    """
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be finite and positive")
    if max_participation_rate is not None and not 0 < max_participation_rate <= 1:
        raise ValueError("max_participation_rate must be in (0, 1]")
    if not np.isfinite(target_tolerance) or not 0 <= target_tolerance < 1:
        raise ValueError("target_tolerance must be in [0, 1)")

    target = pd.to_numeric(weights_today, errors="coerce").fillna(0.0)
    actual = (
        pd.Series(dtype=float)
        if weights_yesterday is None
        else pd.to_numeric(weights_yesterday, errors="coerce").fillna(0.0)
    )
    all_tickers = target.index.union(actual.index)
    w_new = target.reindex(all_tickers, fill_value=0.0)
    w_old = actual.reindex(all_tickers, fill_value=0.0)
    delta = (w_new - w_old)
    delta = delta[delta.abs() > 1e-10]
    if delta.empty:
        return pd.DataFrame(columns=_TRADE_COLUMNS)

    trades = pd.DataFrame(
        {
            "ticker": delta.index,
            "weight_from": w_old.reindex(delta.index, fill_value=0.0).to_numpy(),
            "weight_to": w_new.reindex(delta.index, fill_value=0.0).to_numpy(),
            "delta_weight": delta.to_numpy(),
        }
    )
    trades["side"] = np.where(trades["delta_weight"] > 0, "buy", "sell")
    trades["notional_trade"] = trades["delta_weight"].abs() * capital

    trades["price"] = pd.to_numeric(
        prices_today.reindex(trades["ticker"]).reset_index(drop=True),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    trades["adv_shares"] = pd.to_numeric(
        adv_today.reindex(trades["ticker"]).reset_index(drop=True),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    trades["daily_vol"] = (
        pd.to_numeric(
            vol_today.reindex(trades["ticker"]).reset_index(drop=True),
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.02)
        .clip(lower=0.0)
    )

    valid_price = trades["price"].gt(0) & trades["price"].notna()
    requested = np.zeros(len(trades), dtype=float)
    requested[valid_price.to_numpy()] = np.floor(
        trades.loc[valid_price, "notional_trade"]
        / trades.loc[valid_price, "price"]
    )
    trades["requested_shares"] = requested
    trades["shares"] = requested
    trades["fill_status"] = "filled"
    trades["suppressed_notional"] = 0.0
    trades.loc[~valid_price, ["requested_shares", "shares"]] = 0.0
    trades.loc[~valid_price, "fill_status"] = "rejected_missing_price"

    rounded_zero = valid_price & trades["requested_shares"].le(0)
    trades.loc[rounded_zero, "fill_status"] = "rejected_rounding"

    if max_participation_rate is not None:
        valid_adv = trades["adv_shares"].gt(0) & trades["adv_shares"].ge(min_adv_shares) & trades["adv_shares"].notna()
        no_liquidity = valid_price & ~rounded_zero & ~valid_adv
        trades.loc[no_liquidity, "shares"] = 0.0
        trades.loc[no_liquidity, "fill_status"] = "rejected_no_liquidity"

        eligible = valid_price & ~rounded_zero & valid_adv
        max_shares = np.floor(
            trades.loc[eligible, "adv_shares"] * max_participation_rate
        )
        executed = np.minimum(trades.loc[eligible, "requested_shares"], max_shares)
        trades.loc[eligible, "shares"] = executed
        zero_capacity = eligible & trades["shares"].le(0)
        trades.loc[zero_capacity, "fill_status"] = "rejected_no_liquidity"
        partial = eligible & trades["shares"].gt(0) & (
            trades["shares"] < trades["requested_shares"]
        )
        trades.loc[partial, "fill_status"] = "partial"

    if tradable_today is not None:
        tradable = tradable_today.reindex(trades["ticker"]).fillna(False).to_numpy(dtype=bool)
        trades.loc[~tradable, "shares"] = 0.0
        trades.loc[~tradable, "fill_status"] = "rejected_not_tradable"
    # A zero target, new entry or sign reversal always bypasses the band.
    # Suppressed candidates are audit records, never submitted orders or fills.
    inside_band = (
        trades.weight_to.mul(trades.weight_from).gt(0)
        & trades.delta_weight.abs().le(target_tolerance * trades.weight_to.abs())
    ) if target_tolerance else pd.Series(False, index=trades.index)
    trades.loc[inside_band, "suppressed_notional"] = trades.loc[inside_band, "notional_trade"]
    trades.loc[inside_band, ["requested_shares", "shares", "notional_trade"]] = 0.0
    trades.loc[inside_band, "fill_status"] = "suppressed_tolerance"
    denom = trades["adv_shares"].where(trades["adv_shares"].gt(0))
    trades["requested_participation"] = trades["requested_shares"] / denom
    trades["executed_participation"] = trades["shares"] / denom

    trades["executed_notional"] = (
        trades["shares"] * trades["price"].fillna(0.0)
    )
    direction = np.where(trades["side"].eq("buy"), 1.0, -1.0)
    trades["executed_delta_weight"] = (
        direction * trades["executed_notional"] / capital
    )
    trades["fill_ratio"] = np.where(
        trades["requested_shares"] > 0,
        trades["shares"] / trades["requested_shares"],
        0.0,
    )
    return trades[_TRADE_COLUMNS].reset_index(drop=True)


def _attach_costs(trades: pd.DataFrame, model: CostModel) -> pd.DataFrame:
    """Attach zero costs to rejects and estimated costs to actual fills."""
    result = trades.copy()
    for column in _COST_COLUMNS:
        result[column] = 0.0
    executable = (
        result["shares"].gt(0)
        & result["price"].gt(0)
        & result["price"].notna()
    )
    if executable.any():
        costed = estimate_batch_costs(result.loc[executable, _TRADE_COLUMNS], model)
        result.loc[executable, _COST_COLUMNS] = costed[_COST_COLUMNS].to_numpy()
    return result


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", *_TRADE_COLUMNS, *_COST_COLUMNS, "portfolio_value_before_trade"]
    )


def _empty_pending_orders() -> pd.DataFrame:
    """Return an empty ticker-indexed residual-order book."""
    return pd.DataFrame(
        columns=["side", "target_position_value"],
        index=pd.Index([], name="ticker"),
    )


def _residual_orders_after_fills(
    orders: pd.DataFrame,
    capital: float,
    prior_pending: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the next residual-order book from this attempt's fill results.

    Residuals retain each ticker's dollar position target from the original
    rebalance, rather than refreshing portfolio target weights.  Thus a
    blocked name may be retried without causing unrelated, already-filled
    holdings to be rebalanced after drift.
    """
    if orders.empty:
        return _empty_pending_orders()

    next_rows: list[dict[str, object]] = []
    for row in orders.itertuples(index=False):
        if row.fill_status not in _INCOMPLETE_FILL_STATUSES:
            continue
        if prior_pending is None:
            side = str(row.side)
            target_position_value = float(row.weight_to) * capital
        else:
            target_position_value = float(
                prior_pending.loc[row.ticker, "target_position_value"]
            )
            side = str(prior_pending.loc[row.ticker, "side"])
        next_rows.append(
            {
                "ticker": row.ticker,
                "side": side,
                "target_position_value": target_position_value,
            }
        )
    if not next_rows:
        return _empty_pending_orders()
    return pd.DataFrame(next_rows).set_index("ticker")


def _generate_residual_trades(
    pending: pd.DataFrame,
    actual_weights: pd.Series,
    prices_today: pd.Series,
    adv_today: pd.Series,
    vol_today: pd.Series,
    capital: float,
    max_participation_rate: Optional[float],
    tradable_today: Optional[pd.Series] = None,
    min_adv_shares: float = 0.0,
    target_tolerance: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate orders only for ticker-level residual instructions."""
    tickers = pending.index
    actual = actual_weights.reindex(tickers, fill_value=0.0)
    current_position_value = actual * capital
    target_position_value = pending["target_position_value"].astype(float)
    residual_value = target_position_value - current_position_value
    side = pending["side"]
    if (~side.isin(["buy", "sell"])).any():
        raise ValueError("pending order side must be 'buy' or 'sell'")
    active = (
        (side.eq("buy") & residual_value.gt(1e-8))
        | (side.eq("sell") & residual_value.lt(-1e-8))
    )
    active_pending = pending.loc[active].copy()
    if active_pending.empty:
        return pd.DataFrame(columns=_TRADE_COLUMNS), _empty_pending_orders()
    active_tickers = active_pending.index
    retry_target = (
        active_pending["target_position_value"].astype(float) / capital
    )
    orders = generate_trades(
        weights_today=retry_target,
        weights_yesterday=actual.reindex(active_tickers),
        prices_today=prices_today,
        adv_today=adv_today,
        vol_today=vol_today,
        capital=capital,
        max_participation_rate=max_participation_rate,
        tradable_today=tradable_today,
        min_adv_shares=min_adv_shares,
        target_tolerance=target_tolerance,
    )
    return orders, active_pending


def _allocate_tranche_fills(current, desired, executed):
    """Net opposing sleeve instructions internally, then allocate actual fills.

    Internal transfers have zero external turnover/cost. Unfilled external
    quantities are cancelled at this close; that sleeve revisits them on its
    next scheduled decision. No synthetic fill is credited to total holdings.
    """
    change = desired - current
    buys, sells = change.clip(lower=0), -change.clip(upper=0)
    total_buy, total_sell = buys.sum(), sells.sum()
    cross = np.minimum(total_buy, total_sell)
    internal_buy = buys.mul(cross.div(total_buy.replace(0, np.nan)).fillna(0), axis=1)
    internal_sell = sells.mul(cross.div(total_sell.replace(0, np.nan)).fillna(0), axis=1)
    remaining = change - internal_buy + internal_sell
    net = remaining.sum()
    ratio = executed.reindex(net.index, fill_value=0).div(net.replace(0, np.nan)).fillna(0)
    if ((ratio < -1e-9) | (ratio > 1 + 1e-9)).any():
        raise ValueError("Tranche fill exceeds the net instruction")
    return current + internal_buy - internal_sell + remaining.mul(ratio.clip(0, 1), axis=1)


def _attach_order_sources(orders, previous_target, source, is_rebalance, tranches):
    """Exact cost allocation using aligned directional instruction components."""
    for name in ("signal", "boundary", "drift_resize", "lot_residual"):
        orders[f"source_{name}_fraction"] = 0.0
    if not is_rebalance:
        orders["source_lot_residual_fraction"] = 1.0
        return orders
    labels = orders.ticker.map(source).fillna("signal")
    if tranches > 1:
        for name in ("signal", "boundary", "drift_resize"):
            orders[f"source_{name}_fraction"] = labels.eq(name).astype(float)
        return orders
    prior = orders.ticker.map(previous_target).fillna(0.0)
    direction = np.sign(orders.delta_weight)
    changed = (direction * (orders.weight_to - prior)).clip(lower=0)
    gap = (direction * (prior - orders.weight_from)).clip(lower=0)
    fraction = changed.div((changed + gap).replace(0, np.nan)).fillna(0)
    for name in ("signal", "boundary", "drift_resize"):
        orders[f"source_{name}_fraction"] = fraction * labels.eq(name)
    orders["source_drift_resize_fraction"] += 1 - fraction
    return orders


def simulate_execution(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 1_000_000.0,
    adv_lookback: Optional[int] = None,
    vol_lookback: int = 20,
    return_daily_ledger: bool = False,
    missing_terminal_return: str = "raise",
    target_tolerance: float = 0.0,
    tranche_count: int = 1,
    target_policy=None,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """Simulate fills and PnL under a close-target convention.

    If ``daily_weights`` contains ``is_rebalance``, trades are submitted only
    on flagged dates (plus subsequent retries for unfilled orders).  Without
    that column, the first date and dates where the target vector changes are
    treated as rebalance dates.

    The default return value remains the historical trade DataFrame.  Set
    ``return_daily_ledger=True`` to also receive the authoritative daily PnL
    ledger.  The same ledger is stored in ``trades.attrs["daily_ledger"]`` for
    backward-compatible callers.

    ADV uses positive volumes from tradable rows only, shifted one day, with
    at least max(5, adv_lookback//2) observations. Lookbacks shorter than five
    use a five-day window. With participation enforcement enabled, orders
    also require ADV >= cost_model.min_adv_shares.

    ``target_tolerance`` suppresses same-side target gaps within a relative
    band, including residual retries. Exits/entries/sign changes bypass it.
    With ``tranche_count>1``, day zero seeds all sleeves, then one sleeve is
    refreshed per day using NAV/H. Other sleeves drift with actual fills.
    Daily universe exits apply to all sleeves. Tranche residuals are cancelled
    at each close rather than retried against an aggregate stale dollar target.

    Optional ``target_policy(date, rows, actual_weights, rebalance)`` returns
    (target Series, scalar diagnostic dict). It observes marked ACTUAL holdings
    before sizing orders. A policy may implement diagnose(date, weights) for
    post-fill exposure checks. It must obtain estimates from prior data only.
    """
    if cost_model is None:
        cost_model = CostModel()
    if missing_terminal_return not in {"raise", "total_loss", "zero_return"}:
        raise ValueError("missing_terminal_return must be raise, total_loss, or zero_return")
    if adv_lookback is None:
        adv_lookback = int(cost_model.adv_lookback)
    if adv_lookback <= 0:
        raise ValueError("adv_lookback must be positive")
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("initial_capital must be finite and positive")
    if not np.isfinite(target_tolerance) or not 0 <= target_tolerance < 1:
        raise ValueError("target_tolerance must be in [0, 1)")
    if not isinstance(tranche_count, int) or isinstance(tranche_count, bool) or tranche_count < 1:
        raise ValueError("tranche_count must be a positive integer")
    if target_policy is not None and tranche_count != 1:
        raise ValueError("A dynamic target policy cannot be combined with tranches")
    required_weights = {"date", "ticker", "weight"}
    required_prices = {"date", "ticker", "adj_close"}
    if missing := required_weights.difference(daily_weights.columns):
        raise ValueError(f"daily_weights missing columns: {sorted(missing)}")
    if missing := required_prices.difference(prices.columns):
        raise ValueError(f"prices missing columns: {sorted(missing)}")

    dw = daily_weights.copy()
    execution_columns = ["date", "ticker", "adj_close", "close", "volume", "is_tradable",
                         "total_return", "is_terminal", "permno"]
    px = prices[[c for c in execution_columns if c in prices]].copy()
    dw["date"] = pd.to_datetime(dw["date"])
    px["date"] = pd.to_datetime(px["date"])
    dw["weight"] = pd.to_numeric(dw["weight"], errors="coerce").fillna(0.0)
    if dw.duplicated(["date", "ticker"]).any():
        raise ValueError("daily_weights must have at most one row per date/ticker")
    if px.duplicated(["date", "ticker"]).any():
        raise ValueError("prices must have at most one row per date/ticker")

    px = px.sort_values(["ticker", "date"]).copy()
    px["stock_ret"] = (pd.to_numeric(px["total_return"], errors="coerce") if "total_return" in px
                       else px.groupby("ticker", sort=False)["adj_close"].pct_change(fill_method=None))
    if "is_terminal" not in px:
        px["is_terminal"] = False
    if (px["stock_ret"].dropna() < -1).any():
        raise ValueError("Total returns cannot be below -100%")
    tradable = px["is_tradable"].fillna(False).astype(bool) if "is_tradable" in px else pd.Series(True, index=px.index)
    px["is_tradable"] = tradable
    if "volume" in px:
        volume = pd.to_numeric(px["volume"], errors="coerce")
        volume = volume.where(tradable & volume.gt(0) & np.isfinite(volume))
        min_periods = max(5, adv_lookback // 2)
        px["adv_shares"] = volume.groupby(px["ticker"], sort=False).transform(
            lambda s: s.rolling(max(adv_lookback, min_periods), min_periods=min_periods).mean().shift(1)
        )
    else:
        px["adv_shares"] = np.nan
    px["daily_vol"] = px.groupby("ticker", sort=False)["stock_ret"].transform(
        lambda s: s.rolling(vol_lookback, min_periods=2).std().shift(1)
    )
    px_lookup = px.set_index(["date", "ticker"]).sort_index()
    available_dates = set(px["date"].unique())

    dates = pd.DatetimeIndex(sorted(dw["date"].unique()), name="date")
    if len(dates) == 0:
        empty = _empty_trades()
        ledger = pd.DataFrame(index=dates)
        empty.attrs["daily_ledger"] = ledger
        return (empty, ledger) if return_daily_ledger else empty

    positions = pd.Series(dtype=float)  # signed close-market values
    tranche_positions = pd.DataFrame(index=range(tranche_count), dtype=float)
    last_target: Optional[pd.Series] = None
    pending_orders = _empty_pending_orders()
    nav_net = float(initial_capital)
    nav_gross = float(initial_capital)
    peak_net = nav_net
    peak_gross = nav_gross
    trade_frames: list[pd.DataFrame] = []
    ledger_rows: list[dict] = []
    policy_targets: list[pd.DataFrame] = []
    terminated: set[str] = set()
    weight_groups = {dt: g for dt, g in dw.groupby("date", sort=False)}

    for date_no, dt in enumerate(dates):
        if date_no and date_no % 1000 == 0:
            logger.info("Execution progress: {}/{} sessions, date={}", date_no, len(dates), dt.date())
        group = weight_groups[dt]
        previous_target = last_target if last_target is not None else pd.Series(dtype=float)
        target = group.set_index("ticker")["weight"].sort_index()
        explicit_rebalance = (
            bool(group["is_rebalance"].fillna(False).astype(bool).any())
            if "is_rebalance" in group
            else None
        )
        if explicit_rebalance is None:
            aligned_target = target
            if last_target is None:
                is_rebalance = True
            else:
                idx = target.index.union(last_target.index)
                is_rebalance = not np.allclose(
                    target.reindex(idx, fill_value=0.0),
                    last_target.reindex(idx, fill_value=0.0),
                    atol=1e-12,
                    rtol=0.0,
                )
        else:
            is_rebalance = explicit_rebalance
        last_target = target.copy()

        prev_nav = nav_net
        settlement_value = 0.0
        terminal_imputations = 0
        if dt in available_dates:
            px_today = px_lookup.loc[dt]
            if isinstance(px_today, pd.Series):
                px_today = px_today.to_frame().T
            terminal_today = px_today.index[px_today["is_terminal"].fillna(False).astype(bool)]
            held_terminal = positions.index.intersection(terminal_today)
            day_ret = pd.to_numeric(
                px_today["stock_ret"].reindex(positions.index), errors="coerce"
            )
            missing_terminal = held_terminal.intersection(day_ret.index[day_ret.isna()])
            if len(missing_terminal):
                from quant_platform.core.data.returns import terminal_return_value
                day_ret.loc[missing_terminal] = terminal_return_value(missing_terminal_return)
                terminal_imputations = len(missing_terminal)
            day_ret = day_ret.fillna(0.0)
            pnl_dollar = float((positions * day_ret).sum())
            positions = positions * (1.0 + day_ret)
            if tranche_count > 1:
                tranche_positions = tranche_positions.mul(
                    1 + day_ret.reindex(tranche_positions.columns).fillna(0), axis=1
                ).drop(columns=list(terminal_today), errors="ignore")
            # NAV already contains marked proceeds; removing dollar holdings
            # settles them into implicit cash without a second PnL or a trade.
            settlement_value = float(positions.reindex(held_terminal).sum())
            positions = positions.drop(index=held_terminal)
            terminated.update(terminal_today)
            price_column = "close" if "close" in px_today else "adj_close"
            price_series = pd.to_numeric(px_today[price_column], errors="coerce")
            adv_series = pd.to_numeric(px_today["adv_shares"], errors="coerce")
            vol_series = pd.to_numeric(px_today["daily_vol"], errors="coerce")
            tradable_series = px_today["is_tradable"]
        else:
            pnl_dollar = 0.0
            price_series = pd.Series(dtype=float)
            adv_series = pd.Series(dtype=float)
            vol_series = pd.Series(dtype=float)
            tradable_series = pd.Series(dtype=bool)

        target = target.drop(index=list(terminated), errors="ignore")
        pending_orders = pending_orders.drop(index=list(terminated), errors="ignore")

        gross_ret = pnl_dollar / prev_nav if prev_nav else 0.0
        nav_before_borrow = prev_nav + pnl_dollar
        short_notional = float(-positions[positions < 0].sum())
        borrow_cost = daily_borrow_cost(short_notional, cost_model)
        capital_before_trade = nav_before_borrow - borrow_cost
        if capital_before_trade <= 0:
            raise RuntimeError(f"portfolio NAV became non-positive on {dt.date()}")
        actual_weights = positions / capital_before_trade

        policy_diagnostics = {}
        if target_policy is not None:
            target, policy_diagnostics = target_policy(
                date=dt, rows=group.loc[~group.ticker.isin(terminated)].copy(),
                actual_weights=actual_weights.copy(), rebalance=is_rebalance,
            )
            if not isinstance(target, pd.Series) or target.index.has_duplicates or not np.isfinite(target).all():
                raise ValueError("Dynamic policy must return unique finite target weights")
            target = target.drop(index=list(terminated), errors="ignore")
            last_target = target.copy()
            if is_rebalance:
                policy_targets.append(pd.DataFrame({"date": dt, "ticker": target.index, "weight": target.to_numpy()}))

        suppressed_tranche_notional = 0.0
        suppressed_tranche_count = 0
        if tranche_count > 1:
            if not is_rebalance:
                raise ValueError("Overlapping tranches require daily rebalance instructions")
            columns = tranche_positions.columns.union(target.index)
            tranche_positions = tranche_positions.reindex(columns=columns, fill_value=0.0)
            desired_tranches = tranche_positions.copy()
            active = date_no % tranche_count
            if date_no == 0:
                desired_tranches.loc[:, :] = np.tile(
                    target.reindex(columns, fill_value=0).to_numpy() * capital_before_trade / tranche_count,
                    (tranche_count, 1),
                )
            else:
                desired_tranches.loc[active] = target.reindex(columns, fill_value=0) * capital_before_trade / tranche_count
            if target_tolerance:
                gap = desired_tranches - tranche_positions
                inside = desired_tranches.mul(tranche_positions).gt(0) & gap.abs().le(
                    target_tolerance * desired_tranches.abs()
                )
                suppressed_tranche_notional = float(gap.where(inside, 0).abs().to_numpy().sum())
                suppressed_tranche_count = int(inside.to_numpy().sum())
                desired_tranches = desired_tranches.where(~inside, tranche_positions)
            # Membership exits are mandatory even for an inactive sleeve.
            eligible = set(group.loc[group.eligible.fillna(False), "ticker"]) if "eligible" in group else set(group.ticker)
            desired_tranches.loc[:, ~desired_tranches.columns.isin(eligible)] = 0.0
            target = desired_tranches.sum() / capital_before_trade

        trading_cost = 0.0
        requested_notional = 0.0
        executed_notional = 0.0
        should_submit = is_rebalance or not pending_orders.empty
        if should_submit:
            if is_rebalance:
                # A new target supersedes all stale residual instructions.
                pending_orders = _empty_pending_orders()
                orders = generate_trades(
                    weights_today=target,
                    weights_yesterday=actual_weights,
                    prices_today=price_series,
                    adv_today=adv_series,
                    vol_today=vol_series,
                    capital=capital_before_trade,
                    max_participation_rate=cost_model.max_participation_rate,
                    tradable_today=tradable_series,
                    min_adv_shares=cost_model.min_adv_shares,
                    target_tolerance=target_tolerance if tranche_count == 1 else 0.0,
                )
                prior_pending = None
            else:
                orders, prior_pending = _generate_residual_trades(
                    pending=pending_orders,
                    actual_weights=actual_weights,
                    prices_today=price_series,
                    adv_today=adv_series,
                    vol_today=vol_series,
                    capital=capital_before_trade,
                    max_participation_rate=cost_model.max_participation_rate,
                    tradable_today=tradable_series,
                    min_adv_shares=cost_model.min_adv_shares,
                    target_tolerance=target_tolerance,
                )
            if not orders.empty:
                source = group.set_index("ticker").transition_source if "transition_source" in group else pd.Series(dtype=object)
                orders = _attach_order_sources(orders, previous_target, source, is_rebalance, tranche_count)
                orders = _attach_costs(orders, cost_model)
                orders.insert(0, "date", dt)
                orders["is_rebalance_order"] = is_rebalance
                orders["portfolio_value_before_trade"] = capital_before_trade
                trade_frames.append(orders)
                trading_cost = float(orders["total_cost"].sum())
                requested_notional = float(orders["notional_trade"].sum())
                executed_notional = float(orders["executed_notional"].sum())
                signed_execution = np.where(
                    orders["side"].eq("buy"),
                    orders["executed_notional"],
                    -orders["executed_notional"],
                )
                position_change = pd.Series(
                    signed_execution, index=orders["ticker"]
                ).groupby(level=0).sum()
                if tranche_count > 1:
                    tranche_positions = _allocate_tranche_fills(
                        tranche_positions, desired_tranches, position_change
                    )
                positions = positions.add(position_change, fill_value=0.0)
                positions = positions[positions.abs() > 1e-8]
                pending_orders = _residual_orders_after_fills(
                    orders,
                    capital=capital_before_trade,
                    prior_pending=prior_pending,
                )
            else:
                pending_orders = _empty_pending_orders()
                if tranche_count > 1:
                    tranche_positions = _allocate_tranche_fills(
                        tranche_positions, desired_tranches, pd.Series(dtype=float)
                    )

        cancelled_tranche_residuals = len(pending_orders) if tranche_count > 1 else 0
        if tranche_count > 1:
            pending_orders = _empty_pending_orders()
            np.testing.assert_allclose(
                tranche_positions.sum().reindex(positions.index, fill_value=0), positions,
                rtol=1e-10, atol=1e-7,
            )

        nav_net = capital_before_trade - trading_cost
        if nav_net <= 0:
            raise RuntimeError(f"portfolio NAV became non-positive on {dt.date()}")
        nav_gross *= 1.0 + gross_ret
        peak_net = max(peak_net, nav_net)
        peak_gross = max(peak_gross, nav_gross)
        actual_after = positions / nav_net
        if target_policy is not None and hasattr(target_policy, "diagnose"):
            policy_diagnostics = {**policy_diagnostics, **target_policy.diagnose(dt, actual_after)}
        ledger_rows.append(
            {
                "date": dt,
                "gross_ret": gross_ret,
                "net_ret": nav_net / prev_nav - 1.0,
                "trading_cost_dollar": trading_cost,
                "borrow_cost_dollar": borrow_cost,
                "total_cost_dollar": trading_cost + borrow_cost,
                "trading_cost_bps": trading_cost / prev_nav * 10_000,
                "borrow_cost_bps": borrow_cost / prev_nav * 10_000,
                "requested_turnover": requested_notional / capital_before_trade,
                "executed_turnover": executed_notional / capital_before_trade,
                "one_way_turnover": 0.5 * executed_notional / capital_before_trade,
                "suppressed_notional": suppressed_tranche_notional + (
                    float(orders.suppressed_notional.sum()) if should_submit and not orders.empty else 0.0
                ),
                "suppressed_tranche_candidates": suppressed_tranche_count,
                "cancelled_tranche_residuals": cancelled_tranche_residuals,
                "nav_gross": nav_gross,
                "nav_net": nav_net,
                "drawdown_gross": nav_gross / peak_gross - 1.0,
                "drawdown_net": nav_net / peak_net - 1.0,
                "gross_exposure": float(actual_after.abs().sum()),
                "net_exposure": float(actual_after.sum()),
                "short_exposure": float(-actual_after[actual_after < 0].sum()),
                "is_rebalance": is_rebalance,
                "has_pending_orders": not pending_orders.empty,
                "terminal_settlement_value": settlement_value,
                "terminal_return_imputations": terminal_imputations,
                "missing_terminal_return_policy": missing_terminal_return,
                **{f"risk_{key}": value for key, value in policy_diagnostics.items()},
            }
        )

    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else _empty_trades()
    )
    ledger = pd.DataFrame(ledger_rows).set_index("date")
    if "permno" in px:
        identity = px[["ticker", "permno"]].drop_duplicates()
        if identity["ticker"].duplicated().any():
            raise ValueError("Ticker compatibility key maps to multiple PERMNOs")
        trades["permno"] = trades["ticker"].map(identity.set_index("ticker")["permno"])
    trades.attrs["daily_ledger"] = ledger
    if policy_targets:
        trades.attrs["target_history"] = pd.concat(policy_targets, ignore_index=True)
    if trades.empty:
        logger.warning("No trades generated across the backtest period")
    else:
        logger.info(
            "Execution: {} orders ({} filled/partial) across {} days, "
            "trading cost=${:,.0f}",
            len(trades),
            int(trades["shares"].gt(0).sum()),
            trades["date"].nunique(),
            trades["total_cost"].sum(),
        )
    return (trades, ledger) if return_daily_ledger else trades
