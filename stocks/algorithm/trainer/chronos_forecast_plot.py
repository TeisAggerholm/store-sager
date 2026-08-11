"""Forecast random 11700 val stocks with Chronos-2 and plot median + 80% PI."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from chronos.chronos2 import Chronos2Pipeline

# --- data ------------------------------------------------------------------
PREDICTION_LENGTH = 16
DATA_PATH = "stocks/algorithm/datasets/data/us_stocks/training/11700"
TARGET = "stock"
PAST_COVARIATES = ["volume"]
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
CONTEXT_LENGTH = 1024
HISTORY_PLOT_DAYS = 120  # visual context window before forecast start
NUM_TICKERS = 5  # random tickers to plot sequentially (close each window to continue)

# --- model -----------------------------------------------------------------
# HuggingFace id, or a local artifacts path (finetuned-ckpt / checkpoint-* / run dir).
# Examples:
#   amazon/chronos-2
#   stocks/artifacts/chronos/runs/mock-lora-smoke/finetuned-ckpt
#   stocks/artifacts/chronos/runs/lora-11700-.../checkpoint-1000
MODEL_PATH = "amazon/chronos-2"

STOCKS_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs"
DEFAULT_SAVE_DIR = RUNS_DIR / "forecast-plots"


def _resolve_model_path(model_path: str) -> str:
    """Return a local path if it resolves on disk, otherwise treat as HF model id."""
    candidates = [
        Path(model_path),
        Path.cwd() / model_path,
        STOCKS_DIR / model_path,
        STOCKS_DIR.parent / model_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            # Allow pointing at a run dir: prefer finetuned-ckpt, else latest checkpoint-*.
            if resolved.is_dir() and not (resolved / "config.json").exists():
                finetuned = resolved / "finetuned-ckpt"
                if finetuned.exists():
                    return str(finetuned)
                checkpoints = sorted(
                    resolved.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                if checkpoints:
                    return str(checkpoints[-1])
            return str(resolved)
    return model_path


def _resolve_data_path(data_path: str) -> Path:
    candidates = [
        Path(data_path),
        Path.cwd() / data_path,
        STOCKS_DIR / data_path.removeprefix("stocks/"),
        STOCKS_DIR.parent / data_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Data path not found: {data_path}. "
        "Expected stock_val.csv / volume_val.csv under the 11700 training split."
    )


def _load_val_frames(
    data_dir: Path,
    target: str,
    past_covariates: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    target_df = pd.read_csv(data_dir / f"{target}_val.csv", index_col=0)
    cov_dfs = {
        name: pd.read_csv(data_dir / f"{name}_val.csv", index_col=0)
        for name in past_covariates
    }
    return target_df, cov_dfs


def _aligned_series(
    target_df: pd.DataFrame,
    cov_dfs: dict[str, pd.DataFrame],
    row_idx: int,
) -> tuple[str, pd.Series, dict[str, pd.Series]]:
    ticker = str(target_df.index[row_idx])
    target_row = pd.to_numeric(target_df.iloc[row_idx], errors="coerce")
    cov_rows = {
        name: pd.to_numeric(df.iloc[row_idx], errors="coerce")
        for name, df in cov_dfs.items()
    }

    mask = target_row.notna()
    for cov_row in cov_rows.values():
        mask &= cov_row.notna()

    dates = pd.to_datetime(target_row.index[mask])
    target = target_row[mask].astype(float)
    target.index = dates
    covariates = {}
    for name, cov_row in cov_rows.items():
        series = cov_row[mask].astype(float)
        series.index = dates
        covariates[name] = series
    return ticker, target, covariates


def _eligible_rows(
    target_df: pd.DataFrame,
    cov_dfs: dict[str, pd.DataFrame],
    min_length: int,
) -> list[int]:
    eligible: list[int] = []
    for i in range(len(target_df)):
        _, target, _ = _aligned_series(target_df, cov_dfs, i)
        if len(target) >= min_length:
            eligible.append(i)
    if not eligible:
        raise RuntimeError(f"No val series with length >= {min_length}")
    return eligible


def _row_for_ticker(target_df: pd.DataFrame, ticker: str) -> int:
    if ticker not in target_df.index:
        raise KeyError(f"Ticker {ticker!r} not in val set ({len(target_df)} series)")
    loc = target_df.index.get_loc(ticker)
    if isinstance(loc, slice):
        raise KeyError(f"Ticker {ticker!r} is ambiguous in val index")
    return int(loc)


def _pick_rows(
    target_df: pd.DataFrame,
    cov_dfs: dict[str, pd.DataFrame],
    *,
    min_length: int,
    n: int,
    tickers: list[str] | None,
    seed: int | None,
) -> list[int]:
    if tickers:
        return [_row_for_ticker(target_df, ticker) for ticker in tickers]

    eligible = _eligible_rows(target_df, cov_dfs, min_length)
    k = min(n, len(eligible))
    rng = random.Random(seed)
    return rng.sample(eligible, k=k)


def _plot_forecast(
    *,
    ticker: str,
    context_dates: pd.DatetimeIndex,
    context_target: list[float],
    actual_dates: pd.DatetimeIndex,
    actual_target: list[float],
    forecast_dates: pd.DatetimeIndex,
    p10: list[float],
    p50: list[float],
    p90: list[float],
    context_covariates: dict[str, list[float]],
    actual_covariates: dict[str, list[float]],
    model_label: str,
    save_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    n_cov = len(context_covariates)
    n_rows = 1 + n_cov
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, 3.2 + 2.2 * n_cov),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4] + [1.2] * n_cov},
    )
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(context_dates, context_target, color="tab:blue", label="Price", linewidth=1.6)
    ax.plot(actual_dates, actual_target, color="tab:blue", linewidth=1.6, alpha=0.85)
    ax.plot(forecast_dates, p50, color="magenta", label="Forecast (median)", linewidth=1.8)
    ax.fill_between(
        forecast_dates,
        p10,
        p90,
        color="magenta",
        alpha=0.18,
        label="80% prediction interval",
    )
    forecast_start = forecast_dates[0]
    ax.axvline(forecast_start, color="gray", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} — Chronos-2 ({model_label}), {PREDICTION_LENGTH}-day horizon")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    cov_colors = {"volume": "tab:green"}
    for ax_i, (name, values) in enumerate(context_covariates.items(), start=1):
        color = cov_colors.get(name, "tab:orange")
        axes[ax_i].plot(context_dates, values, color=color, label=name, linewidth=1.4)
        if name in actual_covariates:
            axes[ax_i].plot(
                actual_dates,
                actual_covariates[name],
                color=color,
                linewidth=1.4,
                alpha=0.85,
            )
        axes[ax_i].axvline(forecast_start, color="gray", linestyle="--", linewidth=1.2, alpha=0.85)
        axes[ax_i].set_ylabel(name)
        axes[ax_i].legend(loc="upper left")
        axes[ax_i].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _forecast_and_plot(
    pipeline: Chronos2Pipeline,
    *,
    ticker: str,
    target: pd.Series,
    covariates: dict[str, pd.Series],
    model_label: str,
    save_dir: Path,
    show: bool,
    plot_index: int,
    plot_total: int,
) -> None:
    context_target = target.iloc[:-PREDICTION_LENGTH]
    actual_target = target.iloc[-PREDICTION_LENGTH:]
    context_cov = {name: series.iloc[:-PREDICTION_LENGTH] for name, series in covariates.items()}
    actual_cov = {name: series.iloc[-PREDICTION_LENGTH:] for name, series in covariates.items()}

    # Chronos only needs history up to forecast start; volume is past-only (no future_covariates).
    predict_input = {
        "target": context_target.tolist(),
        "past_covariates": {name: series.tolist() for name, series in context_cov.items()},
    }

    print(f"[{plot_index}/{plot_total}] ticker={ticker} | val length={len(target)}")
    quantiles, _ = pipeline.predict_quantiles(
        [predict_input],
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
        context_length=CONTEXT_LENGTH,
        cross_learning=False,
    )
    fan = quantiles[0][0].detach().cpu().tolist()  # (H, Q)
    p10_idx = QUANTILE_LEVELS.index(0.1)
    p50_idx = QUANTILE_LEVELS.index(0.5)
    p90_idx = QUANTILE_LEVELS.index(0.9)
    p10 = [step[p10_idx] for step in fan]
    p50 = [step[p50_idx] for step in fan]
    p90 = [step[p90_idx] for step in fan]

    plot_context = context_target.iloc[-HISTORY_PLOT_DAYS:]
    plot_context_cov = {
        name: series.iloc[-HISTORY_PLOT_DAYS:].tolist()
        for name, series in context_cov.items()
    }
    save_path = save_dir / f"{ticker}_{PREDICTION_LENGTH}d_{model_label}.png"

    _plot_forecast(
        ticker=ticker,
        context_dates=plot_context.index,
        context_target=plot_context.tolist(),
        actual_dates=actual_target.index,
        actual_target=actual_target.tolist(),
        forecast_dates=actual_target.index,
        p10=p10,
        p50=p50,
        p90=p90,
        context_covariates=plot_context_cov,
        actual_covariates={name: series.tolist() for name, series in actual_cov.items()},
        model_label=model_label,
        save_path=save_path,
        show=show,
    )

    print(
        f"p50 forecast: {[round(x, 3) for x in p50]}\n"
        f"actual:       {[round(x, 3) for x in actual_target.tolist()]}"
    )
    if show and plot_index < plot_total:
        print("Close the plot window to continue to the next ticker...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="HF model id or local artifacts path (default: %(default)s)",
    )
    parser.add_argument("--data-path", default=DATA_PATH, help="11700 training split directory")
    parser.add_argument(
        "-n",
        "--num-tickers",
        type=int,
        default=NUM_TICKERS,
        help="Number of random tickers to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=None,
        help="Specific ticker(s); overrides --num-tickers",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for random ticker pick")
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SAVE_DIR),
        help="Directory for the saved PNG",
    )
    parser.add_argument("--no-show", action="store_true", help="Save only; do not open a window")
    args = parser.parse_args()

    data_dir = _resolve_data_path(args.data_path)
    model_path = _resolve_model_path(args.model_path)
    print(f"Data:  {data_dir}")
    print(f"Model: {model_path}")

    target_df, cov_dfs = _load_val_frames(data_dir, TARGET, PAST_COVARIATES)
    min_length = PREDICTION_LENGTH + 2
    row_indices = _pick_rows(
        target_df,
        cov_dfs,
        min_length=min_length,
        n=args.num_tickers,
        tickers=args.ticker,
        seed=args.seed,
    )
    print(f"Will plot {len(row_indices)} ticker(s)")

    cuda_available = torch.cuda.is_available()
    device_map = "auto" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}; device_map={device_map}")
    pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=device_map)
    print(f"Loaded Chronos-2 with {sum(p.numel() for p in pipeline.model.parameters()):,} parameters")

    model_label = Path(model_path).name if Path(model_path).exists() else model_path
    save_dir = Path(args.save_dir)
    total = len(row_indices)

    for i, row_idx in enumerate(row_indices, start=1):
        ticker, target, covariates = _aligned_series(target_df, cov_dfs, row_idx)
        _forecast_and_plot(
            pipeline,
            ticker=ticker,
            target=target,
            covariates=covariates,
            model_label=model_label,
            save_dir=save_dir,
            show=not args.no_show,
            plot_index=i,
            plot_total=total,
        )


if __name__ == "__main__":
    main()

