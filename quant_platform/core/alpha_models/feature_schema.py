"""Explicit prediction features. Identifiers/provenance are never features."""
BUILTIN_FEATURES = frozenset("""
ABNORMAL_VOLUME ACCRUALS ACF_DECAY_60D ASSET_GROWTH BOOK_TO_MARKET
DOWNVOL_60D EARNINGS_YIELD GROSS_PROFITABILITY HMM_TURBULENCE_PROB HURST_60D
IDIOVOL ILLIQ_AMIHUD KF_NOISE_RATIO KF_TREND_STRENGTH KF_VELOCITY KURT_60D
MOM_12_1M MOM_1M MOM_3M MOM_6M MOM_PATH OU_ZSCORE OVERNIGHT_RET_21D
PIOTROSKI_F8 PRICE_IMPACT REALIZED_JUMP RESID_MOM_12_1M ROLLING_KURT_60D
ROLLING_SKEW_60D SKEW_60D SPECTRAL_ENTROPY_60D SPREAD_HL STR_1M STR_1W
TAIL_RISK TURNOVER_20D TURNOVER_60D TURNOVER_ACCEL VOLUME_CV VOL_20D VOL_60D
VOL_OF_VOL
""".split())

RESERVED = frozenset("""date ticker permno permco gvkey cusip ncusip sic siccd
exchcd source vintage data_source is_point_in_time fundamentals_is_point_in_time
membership_is_point_in_time membership_source available_date publish_date rdq
datadate forward_return target prediction fold label_end is_tradable is_terminal
""".split())


def is_metadata(name: str) -> bool:
    key = name.lower()
    return (key in RESERVED or key.startswith(("label_", "source_", "pit_", "membership_"))
            or key.endswith(("_is_point_in_time", "_available_date", "_vintage", "_source")))
