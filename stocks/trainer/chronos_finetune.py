"""Smoke-test Chronos-2 fine-tuning on tiny mock series."""

from __future__ import annotations

from pathlib import Path

from chronos import Chronos2Pipeline

# --- mock data (same dict shape as predict) ---------------------------------
# Each dict = one series. "target" is what Chronos learns to forecast.
# "past_covariates" are extra history channels (not forecast).
PREDICTION_LENGTH = 2

mock_train_inputs = [
    {
        "target": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0],
        "past_covariates": {"volume": [100.0, 110.0, 105.0, 115.0, 120.0, 118.0]},
    },
    {
        "target": [20.0, 21.0, 22.0, 21.5, 23.0, 24.0],
        "past_covariates": {"volume": [200.0, 205.0, 210.0, 208.0, 212.0, 215.0]},
    },
]

mock_val_inputs = [
    {
        "target": [30.0, 31.0, 32.0, 31.0, 33.0, 34.0],
        "past_covariates": {"volume": [300.0, 310.0, 305.0, 315.0, 320.0, 318.0]},
    },
]

# --- training config ---------------------------------------------------------
LEARNING_RATE = 1e-5  # Chronos recommends ~1e-5 for LoRA (1e-6 for full)
FINETUNE_MODE = "lora"  # needs `peft`; falls back to full if missing
NUM_STEPS = 10  # tiny smoke run — use 1000+ for real training
BATCH_SIZE = 2

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

STOCKS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = STOCKS_DIR / "artifacts" / "chronos" / "runs" / "mock-lora-smoke"


def holdout_metrics(
    pipeline: Chronos2Pipeline,
    val_inputs: list[dict],
    *,
    prediction_length: int,
) -> dict[str, float]:
    """Hold out the last ``prediction_length`` target steps; score p50 vs actuals (MAE / RMSE / MAPE)."""
    actuals: list[float] = []
    preds: list[float] = []

    for row in val_inputs:
        target = list(row["target"])
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
    err = [(a - p) for a, p in zip(actuals, preds)]
    mae = sum(abs(e) for e in err) / n
    rmse = (sum(e * e for e in err) / n) ** 0.5
    mape = sum(abs(a - p) / max(abs(a), 1e-8) for a, p in zip(actuals, preds)) / n * 100
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}


def main() -> None:
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    print("val (base):", holdout_metrics(pipeline, mock_val_inputs, prediction_length=PREDICTION_LENGTH))

    finetuned = pipeline.fit(
        inputs=mock_train_inputs,
        validation_inputs=mock_val_inputs,
        prediction_length=PREDICTION_LENGTH,
        finetune_mode=FINETUNE_MODE,
        lora_config=LORA_CONFIG,
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
    )

    ckpt = OUTPUT_DIR / "finetuned-ckpt"
    print(f"Saved to {ckpt}")
    print("val (finetuned):", holdout_metrics(finetuned, mock_val_inputs, prediction_length=PREDICTION_LENGTH))


if __name__ == "__main__":
    main()
