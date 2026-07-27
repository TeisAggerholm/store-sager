"""Plot metrics produced by chronos_finetune.py, accumulated over time across runs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

STOCKS_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs" / "mock-lora-smoke"
HISTORY_PATH = OUTPUT_DIR / "metrics_history.json"


def load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No metrics history at {path}. Run chronos_finetune.py first to generate it."
        )
    return json.loads(path.read_text())


def plot_history(history: list[dict], save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    latest = history[-1]

    # --- latest run's loss curve ------------------------------------------
    ax = axes[0]
    if latest["train_steps"]:
        ax.plot(latest["train_steps"], latest["train_loss"], marker="o", label="train loss")
    if latest["eval_steps"]:
        ax.plot(latest["eval_steps"], latest["eval_loss"], marker="o", label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Fine-tuning loss (latest run: {latest['timestamp']})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- RMSE over time, across runs --------------------------------------
    ax = axes[1]
    timestamps = [datetime.fromisoformat(e["timestamp"]) for e in history]
    base = [e["base_rmse"] for e in history]
    finetuned = [e["finetuned_rmse"] for e in history]

    ax.plot(timestamps, base, marker="o", label="base")
    ax.plot(timestamps, finetuned, marker="o", label="finetuned")
    ax.set_xlabel("run timestamp")
    ax.set_ylabel("holdout RMSE")
    ax.set_title(f"RMSE over time ({len(history)} runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    history = load_history()
    plot_history(history, save_path=OUTPUT_DIR / "metrics.png")

