"""Plot train/eval loss from a HuggingFace trainer_state.json checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

STOCKS_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs"
RUN_PREFIX = "lora-11700-2026-"


def latest_run_dir(runs_dir: Path = RUNS_DIR) -> Path:
    runs = sorted(
        [p for p in runs_dir.glob(f"{RUN_PREFIX}*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError(f"No {RUN_PREFIX}* run dirs under {runs_dir}")
    return runs[-1]


def latest_trainer_state(output_dir: Path | None = None) -> Path:
    run_dir = output_dir if output_dir is not None else latest_run_dir()
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs under {run_dir}")
    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No trainer_state.json in {checkpoints[-1]}")
    return state_path


def load_loss_curves(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"No trainer state at {path}. Run chronos_finetune.py first.")

    log_history = json.loads(path.read_text())["log_history"]
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    return train_steps, train_loss, eval_steps, eval_loss


def plot_loss_curves(
    train_steps: list[int],
    train_loss: list[float],
    eval_steps: list[int],
    eval_loss: list[float],
    *,
    save_path: Path | None = None,
    title: str = "Train / eval loss",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))

    if train_steps:
        ax.plot(train_steps, train_loss, label="train loss", alpha=0.85)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="eval loss", alpha=0.85)

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()


if __name__ == "__main__":
    run_dir = latest_run_dir()
    state_path = latest_trainer_state(run_dir)
    train_steps, train_loss, eval_steps, eval_loss = load_loss_curves(state_path)
    plot_loss_curves(
        train_steps,
        train_loss,
        eval_steps,
        eval_loss,
        save_path=run_dir / "loss.png",
        title=f"Train / eval loss ({run_dir.name} / {state_path.parent.name})",
    )
