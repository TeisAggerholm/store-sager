"""Overnight Chronos-2 from-scratch training on the 11700 US stocks split."""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
import pandas as pd
import torch
from transformers import AutoConfig

# --- data ------------------------------------------------------------------
PREDICTION_LENGTH = 16
FROM_SCRATCH = True
DATA_PATH = "stocks/algorithm/datasets/data/us_stocks/training/11700"
TARGET = "stock"
PAST_COVARIATES = ["volume"]

# --- training --------------------------------------------------------------
LEARNING_RATE = 1e-4  # from-scratch full train; use ~1e-5 for LoRA on pretrained
FINETUNE_MODE = "lora"  # ignored when FROM_SCRATCH (forced to full)
BATCH_SIZE = 16  # counts target+covariate channels → ~8 ticker-pairs / step
NUM_STEPS = 80_000
CONTEXT_LENGTH = 1024
EVAL_SUBSET = 512
EVAL_SEED = 0

LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 1000

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
RUNS_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs"
OUTPUT_DIR = RUNS_DIR / f"from-scratch-11700-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


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
    target_train_df = pd.read_csv(dic_path + "/" + target + "_train.csv", index_col=0)
    target_val_df = pd.read_csv(dic_path + "/" + target + "_val.csv", index_col=0)

    cov_train_dfs = {name: pd.read_csv(dic_path + "/" + name + "_train.csv", index_col=0) for name in past_covariates}
    cov_val_dfs = {name: pd.read_csv(dic_path + "/" + name + "_val.csv", index_col=0) for name in past_covariates}

    train_input = _build_inputs(target_train_df, cov_train_dfs)
    val_input = _build_inputs(target_val_df, cov_val_dfs)

    return train_input, val_input


def _filter_min_length(inputs: list[dict], min_length: int) -> list[dict]:
    return [row for row in inputs if len(row["target"]) >= min_length]


def _sample_eval_subset(val_inputs: list[dict], k: int, seed: int) -> list[dict]:
    if k >= len(val_inputs):
        return val_inputs
    rng = random.Random(seed)
    indices = rng.sample(range(len(val_inputs)), k=k)
    return [val_inputs[i] for i in indices]


def main() -> None:
    min_train_len = PREDICTION_LENGTH * 2  # matches Chronos default min_past=prediction_length
    print(f"Loading data from {DATA_PATH} ...")
    train_inputs, val_inputs = SS_dataformat_to_chronos(DATA_PATH, TARGET, PAST_COVARIATES)
    train_inputs = _filter_min_length(train_inputs, min_train_len)
    val_inputs = _filter_min_length(val_inputs, PREDICTION_LENGTH + 1)
    val_inputs_eval = _sample_eval_subset(val_inputs, EVAL_SUBSET, EVAL_SEED)

    print(
        f"Train series: {len(train_inputs)} | "
        f"Val series: {len(val_inputs)} | "
        f"Eval subset: {len(val_inputs_eval)}"
    )
    print(
        f"Steps: {NUM_STEPS} | batch_size: {BATCH_SIZE} | "
        f"lr: {LEARNING_RATE} | context_length: {CONTEXT_LENGTH}"
    )

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    device_map = "auto" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}; device={device}")

    if not FROM_SCRATCH:
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device_map)
        finetune_mode = FINETUNE_MODE
        lora_config = LORA_CONFIG if finetune_mode == "lora" else None
        print(f"Loaded Chronos-2 model with {sum(p.numel() for p in pipeline.model.parameters()):,} parameters")
    else:
        config = AutoConfig.from_pretrained("amazon/chronos-2")
        model = Chronos2Model(config).to(device)
        pipeline = Chronos2Pipeline(model=model)
        finetune_mode = "full"
        lora_config = None
        print(f"From-scratch Chronos-2 with {sum(p.numel() for p in pipeline.model.parameters()):,} parameters")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run output dir: {OUTPUT_DIR}")
    finetuned = pipeline.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs_eval,
        prediction_length=PREDICTION_LENGTH,
        finetune_mode=finetune_mode,
        lora_config=lora_config,
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
        output_dir=OUTPUT_DIR,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        log_level="info",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        eval_on_start=True,
        save_steps=SAVE_STEPS,
    )

    ckpt = OUTPUT_DIR / "finetuned-ckpt"
    print(f"Saved to {ckpt}")
    _ = finetuned  # returned pipeline is the fine-tuned model


if __name__ == "__main__":
    main()
