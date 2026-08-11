"""Plot train/eval loss from a HuggingFace trainer_state.json checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

STOCKS_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs" / "from-scratch-11700"


def latest_trainer_state(output_dir: Path = OUTPUT_DIR) -> Path:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs under {output_dir}")
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
    state_path = latest_trainer_state()
    train_steps, train_loss, eval_steps, eval_loss = load_loss_curves(state_path)
    plot_loss_curves(
        train_steps,
        train_loss,
        eval_steps,
        eval_loss,
        save_path=OUTPUT_DIR / "metrics.png",
        title=f"Train / eval loss ({state_path.parent.name})",
    )
