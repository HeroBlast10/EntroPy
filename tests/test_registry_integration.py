"""Integration tests for FactorRegistry.compute_all() and PriceFeatureCache."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.core.signals.registry import FactorRegistry
from quant_platform.core.signals.feature_cache import PriceFeatureCache


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    tickers = ["AAPL", "MSFT", "GOOG"]
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                "date": date,
                "ticker": ticker,
                "open": 100 + np.random.randn(),
                "high": 102 + np.random.randn(),
                "low": 98 + np.random.randn(),
                "close": 100 + np.random.randn(),
                "volume": 1000000 + np.random.randint(-100000, 100000),
                "adj_close": 100 + np.random.randn(),
                "adj_factor": 1.0,
            })
    
    return pd.DataFrame(data)


class TestFactorRegistryComputeAll:
    """Test FactorRegistry.compute_all() with and without cache."""
    
    def test_compute_all_without_cache(self, sample_prices):
        """Test basic compute_all() without cache."""
        registry = FactorRegistry()
        registry.discover()
        
        # Compute a subset of factors
        result = registry.compute_all(
            sample_prices,
            factor_names=["MOM_1M", "VOL_20D"],
            use_cache=False,
        )
        
        # Check structure
        assert "date" in result.columns
        assert "ticker" in result.columns
        assert "MOM_1M" in result.columns
        assert "VOL_20D" in result.columns
        assert len(result) == len(sample_prices)
    
    def test_compute_all_with_cache(self, sample_prices):
        """Test compute_all() with cache enabled."""
        registry = FactorRegistry()
        registry.discover()
        
        # Compute with cache enabled
        result = registry.compute_all(
            sample_prices,
            factor_names=["MOM_1M", "VOL_20D"],
            use_cache=True,
        )
        
        # Check structure (should work even if cache not used by factors)
        assert "date" in result.columns
        assert "ticker" in result.columns
        assert "MOM_1M" in result.columns
        assert "VOL_20D" in result.columns
        assert len(result) == len(sample_prices)
    
    def test_compute_all_empty_factor_list(self, sample_prices):
        """Test that compute_all with empty factor_names computes all factors."""
        registry = FactorRegistry()
        registry.discover()
        
        # Empty list falls back to all factors
        result = registry.compute_all(
            sample_prices,
            factor_names=[],
            use_cache=False,
        )
        
        # Should have computed all discovered factors
        assert len(result.columns) > 2  # More than just date, ticker


class TestPriceFeatureCache:
    """Test PriceFeatureCache feature computation."""
    
    def test_cache_basic_features(self, sample_prices):
        """Test basic feature caching (ret_1d, vol_20d)."""
        cache = PriceFeatureCache(sample_prices)
        
        # Get return feature
        ret_1d = cache.get("ret_1d")
        assert isinstance(ret_1d, pd.Series)
        assert len(ret_1d) == len(sample_prices)
        assert not ret_1d.isna().all()
        
        # Should be cached now
        assert "ret_1d" in cache._cache
        
        # Get volatility feature
        vol_20d = cache.get("vol_20d")
        assert isinstance(vol_20d, pd.Series)
        assert len(vol_20d) == len(sample_prices)
    
    def test_cache_volume_features(self, sample_prices):
        """Test volume feature caching."""
        cache = PriceFeatureCache(sample_prices)
        
        dollar_vol = cache.get("dollar_vol")
        assert isinstance(dollar_vol, pd.Series)
        assert len(dollar_vol) == len(sample_prices)
        
        volume_mean_20 = cache.get("volume_mean_20")
        assert isinstance(volume_mean_20, pd.Series)
        assert len(volume_mean_20) == len(sample_prices)
    
    def test_cache_kalman_features(self, sample_prices):
        """Test Kalman filter feature caching."""
        cache = PriceFeatureCache(sample_prices)
        
        # Get Kalman filtered
        kalman_filtered = cache.get("kalman_filtered")
        assert isinstance(kalman_filtered, pd.Series)
        assert len(kalman_filtered) == len(sample_prices)
        assert not kalman_filtered.isna().all()
        
        # All three should be cached together
        assert "kalman_filtered" in cache._cache
        assert "kalman_velocity" in cache._cache
        assert "kalman_gain" in cache._cache
        
        # Get velocity (should be cached)
        kalman_velocity = cache.get("kalman_velocity")
        assert isinstance(kalman_velocity, pd.Series)
        assert len(kalman_velocity) == len(sample_prices)
    
    def test_cache_unknown_feature(self, sample_prices):
        """Test that unknown features raise ValueError."""
        cache = PriceFeatureCache(sample_prices)
        
        with pytest.raises(ValueError, match="Unknown feature"):
            cache.get("nonexistent_feature")
    
    def test_cache_stats(self, sample_prices):
        """Test cache statistics."""
        cache = PriceFeatureCache(sample_prices)
        
        # Initially empty
        stats = cache.stats()
        assert stats["cached_features"] == 0
        
        # After computing some features
        cache.get("ret_1d")
        cache.get("vol_20d")
        
        stats = cache.stats()
        assert stats["cached_features"] >= 2
        assert stats["cache_size_mb"] > 0
    
    def test_cache_clear(self, sample_prices):
        """Test cache clearing."""
        cache = PriceFeatureCache(sample_prices)
        
        cache.get("ret_1d")
        cache.get("vol_20d")
        assert len(cache._cache) > 0
        
        cache.clear()
        assert len(cache._cache) == 0
