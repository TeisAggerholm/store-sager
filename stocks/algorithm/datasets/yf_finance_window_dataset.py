"""Yahoo Finance OHLCV download and sliding-window :class:`Dataset` (``yf`` = ``import yfinance as yf``).

Uses the ``yfinance`` package. Local YF metadata cache path is
set in :mod:`stocks.algorithm.datasets` via ``yf.set_tz_cache_location`` (:data:`stocks.algorithm.datasets.YF_CACHE_DIR`).

The default :data:`DEFAULT_TICKERS` list is built from **US index ETFs** (e.g. broad market and
growth/tech proxies) plus **large-cap single-name equities** across several sectors — not a single
homogeneous asset class.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "XOM",
    "UNH",
    "COST",
]


def _per_ticker_arrays(
    raw: pd.DataFrame,
    feature_cols: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, pd.DatetimeIndex]]:
    """Split a yfinance frame into float arrays and **aligned** calendar indices per symbol."""
    out: dict[str, np.ndarray] = {}
    dates: dict[str, pd.DatetimeIndex] = {}
    if raw.empty:
        return out, dates

    if isinstance(raw.columns, pd.MultiIndex):
        for sym in raw.columns.get_level_values(0).unique():
            try:
                block = raw[sym][list(feature_cols)]
            except KeyError:
                continue
            block = block.astype(np.float64).dropna(how="any")
            if len(block) < 2:
                continue
            sym_s = str(sym)
            out[sym_s] = block.to_numpy(dtype=np.float32, copy=True)
            dates[sym_s] = pd.DatetimeIndex(pd.to_datetime(block.index))
    else:
        block = raw[list(feature_cols)].astype(np.float64).dropna(how="any")
        if len(block) < 2:
            return out, dates
        out["_"] = block.to_numpy(dtype=np.float32, copy=True)
        dates["_"] = pd.DatetimeIndex(pd.to_datetime(block.index))

    return out, dates


class YFinanceWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sliding (**overlapping**) windows over Yahoo Finance OHLCV — one sample per window per ticker.

    Window starts advance one row at a time (stride ``1``).

    Each ``__getitem__`` returns ``(x, y)``.

    **x** — shape ``(seq_len, n_features)``: OHLCV in time order, newest bar last.

    **y** — shape ``(1,)``: next-bar log return or close ratio vs last close in the window.

    Splits for train/val/test are **not** done here; use :func:`build_yf_window_loaders`.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        *,
        seq_len: int = 20,
        feature_cols: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
        interval: str = "1d",
        auto_adjust: bool = True,
        group_by: str = "ticker",
        progress: bool = False,
        target: Literal["next_log_return", "next_close_ratio"] = "next_log_return",
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be at least 1")
        if "Close" not in feature_cols:
            raise ValueError("feature_cols must include 'Close' for the default targets")

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by=group_by,
            progress=progress,
        )
        self._arrays, self._dates = _per_ticker_arrays(raw, feature_cols)
        self._close_col = feature_cols.index("Close")
        self._seq_len = seq_len
        self._target = target

        self._windows: list[tuple[str, int]] = []
        for sym in sorted(self._arrays.keys()):
            arr = self._arrays[sym]
            n = arr.shape[0]
            for start_row in range(0, max(0, n - seq_len)):
                self._windows.append((sym, start_row))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sym, start_row = self._windows[idx]
        arr = self._arrays[sym]
        window = arr[start_row : start_row + self._seq_len]
        close_last = float(window[-1, self._close_col])
        close_next = float(arr[start_row + self._seq_len, self._close_col])

        # Per-window per-feature z-score. Raw OHLC ($10^2) and Volume ($10^7) have wildly
        # different scales across tickers and time, which destabilises unnormalised FFNs.
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        window_norm = ((window - mean) / std).astype(np.float32, copy=False)

        x = torch.from_numpy(np.ascontiguousarray(window_norm))
        denom = max(close_last, 1e-12)
        if self._target == "next_log_return":
            y = torch.tensor([np.log(close_next / denom)], dtype=torch.float32)
        else:
            y = torch.tensor([close_next / denom], dtype=torch.float32)
        return x, y


def _chronological_split_indices(
    ds: YFinanceWindowDataset,
    *,
    val_fraction: float,
    test_fraction: float,
    include_test: bool,
) -> tuple[list[int], list[int], list[int] | None]:
    """Sort samples by **target bar** timestamp, then contiguous train / val / test index ranges."""
    keyed: list[tuple[pd.Timestamp, str, int]] = []
    for i, (sym, start_row) in enumerate(ds._windows):
        r = start_row + ds._seq_len
        ts = pd.Timestamp(ds._dates[sym][r])
        keyed.append((ts, sym, i))
    keyed.sort(key=lambda t: (t[0], t[1]))
    ordered = [t[2] for t in keyed]
    n = len(ordered)
    if include_test:
        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            rem = n - 1
            n_val = max(1, rem // 2)
            n_test = n - n_train - n_val
        train_idx = ordered[:n_train]
        val_idx = ordered[n_train : n_train + n_val]
        test_idx = ordered[n_train + n_val :]
        return train_idx, val_idx, test_idx

    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError("val_fraction must be in (0, 1) when include_test is False")
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    if n_train < 1:
        raise ValueError("Not enough samples for chronological train/val split.")
    train_idx = ordered[:n_train]
    val_idx = ordered[n_train:]
    return train_idx, val_idx, None


def _window_inputs_overlap(
    ds: YFinanceWindowDataset,
    i: int,
    j: int,
) -> bool:
    """True if windows ``i`` and ``j`` share at least one row on the same ticker (stride-1 sliding)."""
    sym_i, a = ds._windows[i]
    sym_j, b = ds._windows[j]
    if sym_i != sym_j:
        return False
    return abs(a - b) < ds._seq_len


def _purge_overlaps_at_boundaries(
    ds: YFinanceWindowDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None,
) -> tuple[list[int], list[int], list[int] | None]:
    """Drop windows whose **input** bars overlap another split (same symbol), cf. purging in finance ML."""
    val_and_test = val_idx + (test_idx or [])

    new_train = [
        i
        for i in train_idx
        if not any(_window_inputs_overlap(ds, i, j) for j in val_and_test)
    ]

    test_list = test_idx or []
    new_val = [
        i
        for i in val_idx
        if not any(_window_inputs_overlap(ds, i, j) for j in test_list)
    ]

    # Drop test windows that share input rows with *any* val window (including val rows later
    # removed val↔test), so test never overlaps the val time region.
    new_test: list[int] | None
    if test_list:
        new_test = [
            i
            for i in test_list
            if not any(_window_inputs_overlap(ds, i, j) for j in val_idx)
        ]
    else:
        new_test = None

    if len(new_train) < 1:
        raise ValueError(
            "Purging removed all training windows; use a longer history, smaller seq_len, "
            "or smaller val/test fractions."
        )
    if len(new_val) < 1:
        raise ValueError(
            "Purging removed all validation windows; try smaller val_fraction or more data."
        )
    if test_list and new_test is not None and len(new_test) < 1:
        raise ValueError(
            "Purging removed all test windows; try smaller test_fraction or more data."
        )

    return new_train, new_val, new_test


def build_yf_window_loaders(
    *,
    batch_size: int = 32,
    seq_len: int = 20,
    start: str = "2015-01-01",
    end: str | None = None,
    tickers: list[str] | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    shuffle_train: bool = True,
    include_test: bool = True,
    **kwargs: Any,
) -> tuple[DataLoader, DataLoader, DataLoader | None, YFinanceWindowDataset]:
    """Build train / val / (optional) test loaders with a **chronological** split.

    Windows are **overlapping** (stride 1). Samples are sorted by **target bar** time, then split
    into contiguous time blocks (train → val → test).

    **Purging:** any window whose input rows would overlap a window assigned to a **later** split
    (same ticker, ``|Δstart_row| < seq_len``) is dropped from the earlier split, so train/val/test
    do not share bar-level features across boundaries (Lopez de Prado–style purge for sliding
    windows).

    Extra ``kwargs`` are forwarded to :class:`YFinanceWindowDataset`.

    Returns ``(train_loader, val_loader, test_loader, dataset)``. If ``include_test`` is ``False``,
    ``test_loader`` is ``None``. ``dataset`` is the full :class:`YFinanceWindowDataset` backing the
    subsets (useful for metadata, e.g. ticker and dates per window index).
    """
    syms = tickers if tickers is not None else DEFAULT_TICKERS
    ds = YFinanceWindowDataset(syms, start=start, end=end, seq_len=seq_len, **kwargs)
    n = len(ds)
    if n < 3:
        raise ValueError(f"Need at least 3 samples to split; got {n}.")

    train_idx, val_idx, test_idx = _chronological_split_indices(
        ds,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        include_test=include_test,
    )
    train_idx, val_idx, test_idx = _purge_overlaps_at_boundaries(
        ds, train_idx, val_idx, test_idx
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx) if test_idx is not None else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader, ds


if __name__ == "__main__":
    ds = YFinanceWindowDataset(DEFAULT_TICKERS[:3], start="2020-01-01", seq_len=10)
    if len(ds) == 0:
        raise SystemExit(
            "No samples after download (empty dataframe or all NaN). "
            "Check network and that YF cache is writable (see stocks.algorithm.datasets.YF_CACHE_DIR)."
        )
    loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=len(ds) > 1)
    xb, yb = next(iter(loader))
    print("batch x:", xb.shape, "dtype", xb.dtype)
    print("batch y:", yb.shape, "dtype", yb.dtype)
