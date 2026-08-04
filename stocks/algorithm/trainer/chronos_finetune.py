"""Smoke-test Chronos-2 fine-tuning on tiny mock series."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chronos.chronos2 import Chronos2Pipeline
import pandas as pd
import torch

# --- mock data (same dict shape as predict) ---------------------------------
# Each dict = one series. "target" is what Chronos learns to forecast.
# "past_covariates" are extra history channels (not forecast).
PREDICTION_LENGTH = 16

"""
Turn store-sager-stocks data format into chronos
"""
path = "stocks/algorithm/datasets/data/us_stocks/training/32"
target = "stock" # the first always being the target


def _build_inputs(target_df, cov_dfs):
    """Build chronos-format rows, aligning target and covariates on shared non-NaN dates."""
    inputs = []
    for i in range(len(target_df)):
        target_row = pd.to_numeric(target_df.iloc[i], errors="coerce")
        cov_rows = {name: pd.to_numeric(df.iloc[i], errors="coerce") for name, df in cov_dfs.items()}

        mask = target_row.notna()
        for cov_row in cov_rows.values():
            mask &= cov_row.notna()

        inputs.append({
            "target": target_row[mask].tolist(),
            "past_covariates": {name: row[mask].tolist() for name, row in cov_rows.items()},
        })
    return inputs

def SS_dataformat_to_chronos(dic_path, target, past_covariates):
    # load *_train.csv / *_val.csv
    target_train_df = pd.read_csv(dic_path + "/" + target + "_train.csv", index_col=0)
    target_val_df = pd.read_csv(dic_path + "/" + target + "_val.csv", index_col=0)

    cov_train_dfs = {name: pd.read_csv(dic_path + "/" + name + "_train.csv", index_col=0) for name in past_covariates}
    cov_val_dfs = {name: pd.read_csv(dic_path + "/" + name + "_val.csv", index_col=0) for name in past_covariates}

    train_input = _build_inputs(target_train_df, cov_train_dfs)
    val_input = _build_inputs(target_val_df, cov_val_dfs)

    return train_input, val_input


train_inputs, val_inputs = SS_dataformat_to_chronos(path, target, past_covariates=["volume"])

# --- training config ---------------------------------------------------------
LEARNING_RATE = 1e-5  # Chronos recommends ~1e-5 for LoRA (1e-6 for full)
FINETUNE_MODE = "lora"  # needs `peft`; falls back to full if missing
EPOCHS = 100
BATCH_SIZE = 16
trainset_size = len(train_inputs)
NUM_STEPS = (trainset_size // BATCH_SIZE) * EPOCHS  # total update steps

# Same modules Chronos uses by default; dict is passed to peft.LoraConfig(**LORA_CONFIG).
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": [
        "self_attention.q",
        "self_attention.v",
        "self_attention.k",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ],
}

STOCKS_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs" / "mock-lora-smoke"


def holdout_rmse(
    pipeline: Chronos2Pipeline,
    val_inputs: list[dict],
    *,
    prediction_length: int,
) -> float:
    """Hold out the last ``prediction_length`` target steps; RMSE of p50 vs actuals."""
    actuals: list[float] = []
    preds: list[float] = []

    for row in val_inputs:
        target = list(row["target"])
        if len(target) <= prediction_length:
            continue
        history = target[:-prediction_length]
        actuals.extend(target[-prediction_length:])

        inp: dict = {"target": history}
        if "past_covariates" in row:
            inp["past_covariates"] = {
                name: list(values)[:-prediction_length]
                for name, values in row["past_covariates"].items()
            }

        _, medians = pipeline.predict_quantiles(
            [inp],
            prediction_length=prediction_length,
            quantile_levels=[0.5],
        )
        preds.extend(medians[0][0].tolist())

    n = len(actuals)
    mse = sum((a - p) ** 2 for a, p in zip(actuals, preds)) / n
    return mse ** 0.5


def main() -> None:
    # prefer GPU when available; fall back to CPU
    cuda_available = torch.cuda.is_available()
    device_map = "auto" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}; loading model with device_map={device_map}")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device_map)
    print(f"Loaded Chronos-2 model with {sum(p.numel() for p in pipeline.model.parameters()):,} parameters")
    #base_rmse = holdout_rmse(pipeline, val_inputs, prediction_length=PREDICTION_LENGTH)
    #print(f"val RMSE (base):      {base_rmse:.4f}")

    finetuned = pipeline.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs,
        prediction_length=PREDICTION_LENGTH,
        finetune_mode=FINETUNE_MODE,
        lora_config=LORA_CONFIG,
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,

         #Logging / eval / save controls:
        logging_strategy="steps",     # "steps" or "epoch" or "no"
        logging_steps=1,            # when strategy="steps", how often to log
        logging_first_step=True,     # log the very first step
        log_level="info",            # optional: "debug"|"info"|"warning" etc.

        eval_strategy="steps",
        eval_steps=1,  # ~0.25 epoch for our mock data
        eval_on_start=True, 

    )

    ckpt = OUTPUT_DIR / "finetuned-ckpt"
    print(f"Saved to {ckpt}")
    #finetuned_rmse = holdout_rmse(finetuned, val_inputs, prediction_length=PREDICTION_LENGTH)
    #print(f"val RMSE (finetuned): {finetuned_rmse:.4f}")

    #save_metrics(base_rmse, finetuned_rmse)


def save_metrics(base_rmse: float, finetuned_rmse: float) -> None:
    """Collect train/eval loss curves (from the Trainer's checkpoint) plus RMSE
    before/after fine-tuning, and append them as a new run entry to
    ``OUTPUT_DIR/metrics_history.json`` so metrics accumulate across runs over time."""
    checkpoints = sorted(
        OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    log_history: list[dict] = []
    if checkpoints:
        state_path = checkpoints[-1] / "trainer_state.json"
        if state_path.exists():
            log_history = json.loads(state_path.read_text())["log_history"]

    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    run_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_steps": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "finetune_mode": FINETUNE_MODE,
        "base_rmse": base_rmse,
        "finetuned_rmse": finetuned_rmse,
        "train_steps": train_steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUT_DIR / "metrics_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(run_entry)
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Appended run metrics to {history_path} ({len(history)} runs recorded)")


if __name__ == "__main__":
    main()
