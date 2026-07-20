"""Stock dataset loaders. On-disk files live under :data:`DATA_DIR` (same idea as ``bproj/datasets``)."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yfinance as yf

_BASE = Path(__file__).resolve().parent
DATA_DIR = _BASE / "data"
# Yahoo Finance — yfinance stores tz / cookie / ISIN metadata here (``import yfinance as yf`` is the usual alias).
YF_CACHE_DIR = DATA_DIR / "yf"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE_DIR))

if TYPE_CHECKING:
    from .yf_finance_window_dataset import (
        DEFAULT_TICKERS,
        YFinanceWindowDataset,
        build_yf_window_loaders,
    )

__all__ = [
    "DATA_DIR",
    "YF_CACHE_DIR",
    "DEFAULT_TICKERS",
    "YFinanceWindowDataset",
    "build_yf_window_loaders",
]


def __getattr__(name: str) -> Any:
    if name in ("DEFAULT_TICKERS", "YFinanceWindowDataset", "build_yf_window_loaders"):
        mod = import_module(".yf_finance_window_dataset", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
