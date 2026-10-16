# MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Getting started

Clone the repository **first**, then create the virtual environment **inside** the cloned folder:
`requirements.txt` installs this project's own `mlops` package (`-e .`), so the code must already be
there, and `.venv/` is ignored by git. Commands below are for Windows PowerShell; the macOS/Linux
equivalent is shown where it differs.

### 1. Check the prerequisites (once per computer)

```powershell
git --version
python --version      # 3.10 or newer (the course was tested with 3.13)
```

### 2. Clone the repository

Open a terminal in the folder where you keep your projects (any folder works) and run:

```powershell
git clone https://github.com/ITESM-MNA/MLOps-2021.git
cd MLOps-2021
```

From here on, every command runs from this repository folder (`MLOps-2021`, the project root) and
all paths are relative to it.

### 3. Create and activate the virtual environment (inside the repo folder)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
```

The prompt now starts with `(.venv)`. If PowerShell blocks the script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

### 4. Install the dependencies

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

This also installs the `mlops` package in editable mode, so code changes apply immediately.

### 5. Check the setup

```powershell
python -m pytest -q        # expected: 3 passed
```

### 6. Start the local MLflow tracking server (terminal 1)

All experiments are tracked on a local MLflow server. Open a **second terminal** (terminal 1 from
now on), go to the repo folder, activate `.venv` and leave the server running while you work:

```powershell
cd MLOps-2021                        # the repository folder, from where you cloned it
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```

The server stores runs in `mlflow.db` and files in `mlartifacts/` (both ignored by git). Open
http://127.0.0.1:5000 to see the experiments.

### 7. Set `MLFLOW_TRACKING_URI` manually (terminal 2)

In the terminal where you will work (terminal 2), point MLflow to the server:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"     # macOS/Linux: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo $env:MLFLOW_TRACKING_URI                           # check it; macOS/Linux: echo $MLFLOW_TRACKING_URI
```

> **This variable is set by hand.** It is not stored in `.venv` nor in the repository, and it only
> lasts while that terminal is open. Set it again in **every new terminal** before using
> `mlflow run`. The `mlflow run` command creates the run before any project code executes, so it
> cannot read the address from `mlops/config.py`; if the variable is missing, the run stops with the
> message *`mlflow run` is not using the MLflow server*. Python code (`python -m mlops...`,
> `scripts/run_project.py` and the notebooks) already reads the address from `mlops/config.py`.

### 8. Run the notebooks (1.0 → 2.1)

```powershell
jupyter lab
```

Open `notebooks/` and run them in order (see *Course notebooks* below). In VS Code, select the
`.venv` interpreter as the notebook kernel instead.

### 9. Run the MLflow project (terminal 2, from the repo root)

```powershell
python scripts/prepare_data.py
mlflow run . --env-manager local --experiment-name wine-quality
mlflow run . -P alpha=0.01 --env-manager local --experiment-name wine-quality
mlflow run . -e workflow --env-manager local --experiment-name wine-quality
```

### 10. See the results

Open http://127.0.0.1:5000 (the server from step 6), select runs in the `wine-quality` experiment
and click **Compare**. The cleaning run from notebook 1.4 is in the `wine-quality-data-cleaning`
experiment.

### Every new session

Terminal 1 (MLflow server, leave it running):

```powershell
cd MLOps-2021                        # the repository folder, from where you cloned it
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```

Terminal 2 (your work):

```powershell
cd MLOps-2021                        # the repository folder, from where you cloned it
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"     # macOS/Linux: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
git pull                             # get the latest course material
```

If `git pull` changes `requirements.txt`, run `pip install -r requirements.txt` again.

**Tips**

- The `make ...` shortcuts in this README need `make`, which Windows does not include by
  default; the commands above do the same without it.
- Start the MLflow server from the repo root, so it always uses the same `mlflow.db`.
- If a script says *MLflow tracking server not reachable*, the server from step 6 is not running.
  If `mlflow run` says it *is not using the MLflow server*, set `MLFLOW_TRACKING_URI` (step 7).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── MLproject          <- MLflow Project: entry points that `mlflow run` can execute
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling (versioned with MLflow).
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops and configuration for tools like black and pytest
│
├── python_env.yaml    <- Python version and libraries for `mlflow run` without `--env-manager local`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── scripts            <- prepare_data.py (train/test split) and run_project.py (run the project from Python)
│
├── setup.cfg          <- Configuration file for flake8
│
├── tests              <- Unit tests (pytest)
│
└── mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Step "load": prepares the cleaned dataset (data/interim)
    │
    ├── features.py             <- Step "features": creates features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations
    │
    ├── tracking.py             <- MLflow dataset tracking helpers (log_input)
    │
    ├── transformers.py         <- OutlierClipper (custom scikit-learn transformer)
    │
    └── workflow.py             <- Chains the steps: load -> features -> train
```

## Course notebooks

The notebooks follow the cookiecutter naming convention `<number>-<initials>-<description>.ipynb`
and read/write data only through the paths in `mlops/config.py`. Run them in order:

| Notebook | Reads | Writes |
|---|---|---|
| `1.0-grh-get-wine-data` | UCI `wine+quality.zip` | `data/raw/wine_quality_original.csv` (white wine) |
| `1.1-grh-intro-numpy-pandas` | `data/raw/wine_quality_original.csv` | – |
| `1.2-grh-descriptive-statistics` | `data/raw/wine_quality_original.csv` | – |
| `1.3-grh-wine-eda` | `data/raw/wine_quality_original.csv` | `data/interim/wine_quality_original_no_duplicates.csv` |
| `1.4-grh-wine-data-cleaning` | `data/raw/wine_quality_modified.csv` (+ the files above) | `data/processed/wine_quality_cleaned_dropna.csv`, `data/processed/wine_quality_cleaned_imputed.csv`, MLflow run |
| `2.0-grh-messy-notebook` | `data/processed/wine_quality_cleaned_dropna.csv` | `reports/figures/preds.png` (the "before refactoring" example) |
| `2.1-grh-initial-eda` | `data/processed/wine_quality_cleaned_dropna.csv` (through the `mlops` package) | – |

See *Getting started* above for the environment setup.

## MLflow Projects: experiment and model tracking

The project predicts the **quality of a white wine** (`quality`, from 3 to 9) from its chemical
measurements, starting from the cleaned dataset produced by notebook 1.4
(`data/processed/wine_quality_cleaned_dropna.csv`). The model is the pipeline
`OutlierClipper` → `StandardScaler` → `ElasticNet(alpha)`: the larger `alpha`, the stronger the
regularization and the simpler the model.

All runs are tracked on the **local MLflow tracking server** (`MLFLOW_TRACKING_URI` in
`mlops/config.py`, default `http://127.0.0.1:5000`; notebook 1.4 logs there too). Start it first,
from the project root and in its own terminal, and point the terminal you run from to it:

```powershell
mlflow server --host 127.0.0.1 --port 5000             # terminal 1 (or: make mlflow-server)
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"     # terminal 2, set manually (Getting started, step 7); macOS/Linux: export MLFLOW_TRACKING_URI=...
```

`mlflow run` needs the environment variable because it creates the run before any project code
runs. Python code (`python -m mlops...`, `scripts/run_project.py`, the notebooks) reads the URI
from `mlops/config.py`. To use another server, set `MLFLOW_TRACKING_URI` in the `.env` file.

### 1. Prepare the data

```bash
python scripts/prepare_data.py   # or: make data
```

It writes `data/processed/wine_quality_train.csv` and `wine_quality_test.csv` (80/20 split).

### 2. Run the project

```bash
mlflow run . --env-manager local --experiment-name wine-quality
```

MLflow reads `MLproject`, takes the `main` entry point and runs
`python -m mlops.modeling.train --data data/processed/wine_quality_train.csv --alpha 0.5`.
It should end with `=== Run (ID '...') succeeded ===`.

Try other values of `alpha` to compare them later:

```bash
mlflow run . -P alpha=0.3 --env-manager local --experiment-name wine-quality
mlflow run . -P alpha=0.01 --env-manager local --experiment-name wine-quality
```

### 3. See the results

Open http://127.0.0.1:5000, select several runs of the `wine-quality` experiment and click
**Compare** to see how `rmse`, `mae` and `r2` change with `alpha`. Each run also lists the datasets
it used (see *Dataset tracking with MLflow* below). The browser does not need to be open while the
runs execute, but the server must be running.

### 4. Other entry points

| Command | What it does |
| --- | --- |
| `mlflow run . -e workflow --env-manager local --experiment-name wine-quality` | Runs the 3 steps `load` → `features` → `train`, each as its own run, grouped under a parent run |
| `mlflow run . -e predict -P model_uri=runs:/<run_id>/model --env-manager local --experiment-name wine-quality` | Predicts with a trained model and saves `reports/predictions.csv` |
| `python scripts/run_project.py` | Runs the project from Python instead of the terminal |
| `python -m mlops.plots` | Saves `reports/figures/alcohol_by_quality.png` |
| `make mlflow-run ALPHA=0.3`, `make workflow`, `make test`, `make lint` | Shortcuts for the commands above, the unit tests and the linters |

The `<run_id>` is printed at the end of training (`Model URI: runs:/.../model`).

The server's `mlflow.db` and `mlartifacts/`, `data/interim/wine_quality_interim.csv`,
`data/processed/wine_quality_features.parquet` and `reports/predictions.csv` are regenerated by
the runs and are ignored by git.

## Dataset tracking with MLflow

This project does **not** use DVC. Dataset storage is the local folder `data/processed`
(`DATASET_STORAGE_DIR` in `mlops/config.py`), and the datasets saved there are versioned with
[MLflow dataset tracking](https://mlflow.org/docs/latest/ml/dataset/): each run records which
dataset it used (name, source, digest, schema and profile), so every model can be traced back
to the exact data it was trained on.

```python
import mlflow
import pandas as pd

from mlops.config import DATASET_STORAGE_DIR

path = DATASET_STORAGE_DIR / "wine_quality_cleaned_dropna.csv"
df = pd.read_csv(path)
dataset = mlflow.data.from_pandas(
    df, source=str(path), name="wine_quality_cleaned_dropna", targets="quality"
)

with mlflow.start_run():
    mlflow.log_input(dataset, context="training")
```

The MLflow server's local store (`mlflow.db`, `mlartifacts/`, and `mlruns/` if used) is ignored by git.

--------

