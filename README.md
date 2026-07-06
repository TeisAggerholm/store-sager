# Store sager

*Der var engang, hvor science og matematik hørte hjemme bag universiteternes tykke mure. En lille gruppe dedikerede nørder sad bøjet over tavler og notesbøger, drevet af nysgerrighed, koffein og en stille ambition om at forstå verden lidt bedre end i går. Forskning var noget, der foregik mellem forelæsninger, i sene nattetimer og ved whiteboards fyldt med halvviskede ligninger.*

*Men den tid er forbi.*

*I dag er videnskaben rykket ud af auditorierne og ind på internettets markedspladser. Modeller handles, trænes og forbedres i åbne netværk, hvor ideer konkurrerer i realtid. Det går stærkt. Og det går godt.*

*Ikke i et sterilt laboratorium, men over øl — og flere jägerbombs, end nogen nok vil indrømme — på Caféen begyndte det. En håndfuld ligesindede nørder. De søgte ikke bare svar – de søgte en udfordring. Noget hver især kunne kaste sig over, og som sammen kunne blive til noget, ingen af dem kunne have skabt alene.*

*Det her repo fortæller historien om, hvordan det hele begyndte.*

---

# Python environment

Dependencies and the installable packages (`stocks`, `poker`) are managed with **[uv](https://docs.astral.sh/uv/)** — a fast Python package manager that reads `pyproject.toml` and `uv.lock`.

## Setup

From the repo root:

```bash
uv sync
```

Creates `.venv`, installs locked dependencies (including PyTorch), and installs this project in editable mode so imports like `from stocks.models...` work.

## Run

Use `uv run` so you do not need to activate the virtualenv:

```bash
# Console scripts (see [project.scripts] in pyproject.toml)
uv run mc-poker

# Python modules / scripts
uv run python stocks/models/timeseries_ffn.py
uv run python -c "import torch; print(torch.__version__)"
```

Alternatively, activate the venv once: `source .venv/bin/activate`, then run `mc-poker` or `python ...` directly.

## Add a dependency

```bash
uv add package-name          # add to pyproject.toml and update uv.lock
uv add --dev package-name    # optional / dev dependency
```

Then commit the updated `pyproject.toml` and `uv.lock`.

---

# Poker

Monte Carlo simulation for **all-in equity**: given hero hole cards, a number of opponents, and optionally known board cards, it estimates win / tie / loss probabilities by sampling many random runouts.

## Entry points

| What | Where |
|------|--------|
| Implementation | `poker/mc_poker.py` — `main()` runs the simulation. |
| CLI | Console script **`mc-poker`** (see `[project.scripts]` in `pyproject.toml`). |

## Run

After installing the project (e.g. `uv sync`):

```bash
uv run mc-poker
```

If your virtualenv is active, you can call `mc-poker` directly (it is installed on `PATH` like any other console script).

## Configuration

There is no CLI flags file: you edit the constants at the top of `poker/mc_poker.py` (hero cards, number of opponents, sample count, known board cards), then run **`mc-poker`** again.

For more detail on card notation and behaviour, see `poker/poker.md`.
