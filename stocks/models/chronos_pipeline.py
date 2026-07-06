from chronos import Chronos2Pipeline

chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


if __name__ == "__main__":
    target_history = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
    forecast_steps = 5
    quantile_levels = [0.1, 0.5, 0.9]  # choose your own; predict_quantiles() picks these (predict() returns all model quantiles)

    covariate_1_history = [x * 1.2 + 5 for x in target_history]  # same length as target
    last_covariate = covariate_1_history[-1]
    covariate_1_future = [last_covariate + i for i in range(1, forecast_steps + 1)]  # length = forecast_steps

    # predict() / predict_quantiles() take a list of input dicts — one dict per time series.
    #
    # Each dict can have:
    #   "target" (required)           — what Chronos forecasts (1-d array, or 2-d for multivariate)
    #   "past_covariates" (optional)  — extra history only; same length as target; not forecast
    #   "future_covariates" (optional) — known future covariate values; length = prediction_length
    #                                    keys must also appear in past_covariates
    #
    # cross_learning: shares information across *multiple series in the batch* (multiple dicts in the list),
    # e.g. forecasting AAPL + MSFT together. It does NOT control whether covariates are used — they always
    # are when provided. Leave False for a single series; try True when passing many related series at once.

    quantiles, _ = chronos_pipeline.predict_quantiles(
        [{
            "target": target_history,
            "past_covariates": {"covariate_1": covariate_1_history},
            "future_covariates": {"covariate_1": covariate_1_future},
        }],
        prediction_length=forecast_steps,
        quantile_levels=quantile_levels,
        cross_learning=False,
    )

    # Indexing: quantiles[input_idx][variate_idx][step][quantile_idx]
    #   [0]     first dict in the list (one series here)
    #   [0]     first target variate (univariate → only index 0)
    #   [step]  forecast step 0..forecast_steps-1
    #   [q]     position in quantile_levels (e.g. 1 → 0.5 when levels are [0.1, 0.5, 0.9])
    fan = quantiles[0][0].tolist()  # (forecast_steps, len(quantile_levels))
    p50_idx = quantile_levels.index(0.5)

    print(f"history (last): {target_history[-1]}")
    print(f"p50 forecast:   {[round(step[p50_idx], 2) for step in fan]}")
    print(f"step 1 fan:     p10={fan[0][0]:.2f}  p50={fan[0][p50_idx]:.2f}  p90={fan[0][2]:.2f}")
