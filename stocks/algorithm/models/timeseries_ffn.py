"""
Shared feed-forward stack at each time step on a **fixed-length** padded series, then a single
linear from the **flattened** sequence of hidden states
``(max_seq_len * hidden_last) → output_dim``.

See also: `GeeksforGeeks — Feedforward neural network <https://www.geeksforgeeks.org/deep-learning/feedforward-neural-network/>`_.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeSeriesFFN(nn.Module):
    """FFN shared across time (applied independently at each of ``max_seq_len`` positions).

    **Input:** ``(batch, L, n_features)`` or ``(L, n_features)`` with ``L ≤ max_seq_len``. If
    ``L > max_seq_len``, only the **last** ``max_seq_len`` rows are kept (most recent bars).

    **Padding:** shorter sequences are **left-padded with zeros** to length ``max_seq_len``, so real
    bars sit at the **end**; index ``max_seq_len - 1`` is always the latest bar.

    **Output:** ``(batch, output_dim)`` — after the FFN, hidden states
    ``(batch, max_seq_len, hidden_dims[-1])`` are flattened and passed through
    ``Linear(max_seq_len * hidden_dims[-1], output_dim)``.
    """

    def __init__(
        self,
        n_features: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 64),
        max_seq_len: int = 256,
        output_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        if output_dim < 1:
            raise ValueError("output_dim must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.n_features = n_features
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_d = n_features
        for i, out_d in enumerate(hidden_dims):
            layers.append(nn.Linear(in_d, out_d))
            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_d = out_d
        self.ffn = nn.Sequential(*layers)
        self.out_proj = nn.Linear(max_seq_len * hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, output_dim)`` from flattened FFN outputs over all timesteps."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"x must be 2D or 3D, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"last dim must be n_features={self.n_features}, got {x.shape[-1]}"
            )

        b, seq_len, f = x.shape
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len :, :]
            seq_len = self.max_seq_len

        pad = self.max_seq_len - seq_len
        if pad > 0:
            x = torch.cat(
                [x.new_zeros(b, pad, f), x],
                dim=1,
            )

        h = self.ffn(x)
        return self.out_proj(h.flatten(1, 2))


if __name__ == "__main__":
    import math
    import random
    from datetime import datetime
    from pathlib import Path

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from torch.utils.data import Subset
    from tqdm import tqdm

    from stocks.algorithm.datasets.yf_finance_window_dataset import build_yf_window_loaders

    seq_len = 20
    n_features = 5
    batch_size = 64
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Symbol for test-period return/close plots. Must appear in the test split after purge,
    # unless ``None`` — then a random test-split ticker is chosen.
    ticker: str | None = "AAPL"

    train_loader, val_loader, test_loader, ds = build_yf_window_loaders(
        batch_size=batch_size,
        seq_len=seq_len,
        start="2018-01-01",
        include_test=True,
        progress=False,
    )
    assert test_loader is not None

    model = TimeSeriesFFN(n_features, hidden_dims=(64, 32), max_seq_len=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in tqdm(range(epochs), desc="epoch"):
        model.train()
        train_loss = 0.0
        nt = 0
        for xb, yb in tqdm(
            train_loader,
            desc=f"train {epoch + 1}/{epochs}",
            leave=False,
        ):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            nt += 1
        train_loss /= max(nt, 1)

        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
                nv += 1
        val_loss /= max(nv, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Test-period plots: actual vs predicted next-bar log return (chronological).
    test_ds = test_loader.dataset
    assert isinstance(test_ds, Subset)
    syms_in_test = {ds._windows[i][0] for i in test_ds.indices}
    if not syms_in_test:
        raise ValueError("Test split has no samples.")
    if ticker is None:
        pick_sym = random.choice(sorted(syms_in_test))
    elif ticker not in syms_in_test:
        raise ValueError(
            f"ticker={ticker!r} has no windows in the test split after purge. "
            f"Try one of: {sorted(syms_in_test)}"
        )
    else:
        pick_sym = ticker
    keyed: list[tuple[object, int]] = []
    for i in test_ds.indices:
        sym, start_row = ds._windows[i]
        if sym != pick_sym:
            continue
        r = start_row + ds._seq_len
        ts = ds._dates[sym][r]
        keyed.append((ts, i))
    keyed.sort(key=lambda t: t[0])
    if keyed:
        assert {ds._windows[i][0] for _, i in keyed} == {pick_sym}

    model.eval()
    actual: list[float] = []
    predicted: list[float] = []
    actual_close: list[float] = []
    curve_dates: list[object] = []
    close_col = ds._close_col
    with torch.no_grad():
        for ts, i in keyed:
            x, y = ds[i]
            sym_w, start_row = ds._windows[i]
            assert sym_w == pick_sym
            r = start_row + ds._seq_len
            arr = ds._arrays[sym_w]
            p = model(x.unsqueeze(0).to(device)).cpu().numpy().reshape(-1)[0]
            actual.append(float(y.reshape(-1)[0].item()))
            predicted.append(float(p))
            actual_close.append(float(arr[r, close_col]))
            curve_dates.append(ts)

    # Close on each target date from predicted log returns: compound from first window's pre-target close.
    pred_close: list[float] = []
    if keyed:
        _, i0 = keyed[0]
        sym0, sr0 = ds._windows[i0]
        assert sym0 == pick_sym
        r0 = sr0 + ds._seq_len
        prev = float(ds._arrays[sym0][r0 - 1, close_col])
        for pr in predicted:
            prev *= math.exp(pr)
            pred_close.append(prev)

    xs = list(range(1, epochs + 1))
    fig, (ax_loss, ax_ret, ax_price) = plt.subplots(
        3, 1, figsize=(8, 10), height_ratios=[1, 1.15, 1.15]
    )
    ax_loss.plot(xs, train_losses, label="train")
    ax_loss.plot(xs, val_losses, label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training / validation loss (all tickers, log scale)")
    ax_loss.legend()

    ax_ret.plot(
        curve_dates,
        actual,
        label=f"{pick_sym}: actual next log return",
        alpha=0.85,
    )
    ax_ret.plot(
        curve_dates,
        predicted,
        label=f"{pick_sym}: predicted next log return",
        alpha=0.85,
    )
    ax_ret.set_xlabel("target bar date")
    ax_ret.set_ylabel("next-bar log return")
    ax_ret.set_title(f"{pick_sym} — next-bar log return (test)")
    ax_ret.legend()
    ax_ret.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax_price.plot(
        curve_dates,
        actual_close,
        label=f"{pick_sym}: actual close",
        alpha=0.85,
    )
    ax_price.plot(
        curve_dates,
        pred_close,
        label=f"{pick_sym}: predicted close (compounded)",
        alpha=0.85,
    )
    ax_price.set_xlabel("target bar date")
    ax_price.set_ylabel("close")
    ax_price.set_title(f"{pick_sym} — close (test)")
    ax_price.legend()
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()

    fig.tight_layout()

    runs_dir = Path(__file__).resolve().parents[2] / "artifacts" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ASCII-safe filename (ticker is already a valid symbol string)
    save_path = runs_dir / f"timeseries_ffn_{pick_sym}_{timestamp}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.show()
