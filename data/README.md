# Data

| Folder | Contents | In git? |
|---|---|---|
| `raw/` | The original, immutable data: the UCI `wine+quality.zip`, the white-wine `wine_quality_original.csv`, and the deliberately corrupted `wine_quality_modified.csv` (+ its `_metadata.json`) used for the cleaning exercise. | **Yes** |
| `interim/` | Intermediate data: `wine_quality_original_no_duplicates.csv` (notebook 1.3). | **Yes** |
| `processed/` | Final, model-ready datasets: `wine_quality_cleaned_dropna.csv` and `wine_quality_cleaned_imputed.csv` (notebook 1.4). | **Yes** |
| `external/` | Data from third-party sources. | No (only `.gitkeep`) |

## Dataset storage and tracking

This project does **not** use DVC. Datasets in `data/processed/` (`DATASET_STORAGE_DIR` in
`mlops/config.py`) are tracked with
[MLflow dataset tracking](https://mlflow.org/docs/latest/ml/dataset/), and because they are very
small they are also committed to this repository (as are `data/raw` and `data/interim`).
Notebook `1.4-grh-wine-data-cleaning` logs its input and output datasets to MLflow automatically.

MLflow does not store a copy of the data. For each run it records the dataset's name, digest
(a fingerprint of the contents), schema, profile and **source**, which is the location the data
came from. The data itself must live wherever that `source` points.

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

## ⚠️ Larger or real-world datasets

Keeping data in git only works because these files are tiny. For larger, sensitive or
frequently changing datasets, **do not commit them**. Store them somewhere else and make the
data `source` point there, for example:

- Cloud object storage: `s3://bucket/path/data.csv`, `gs://bucket/path/data.csv`,
  `abfss://container@account.dfs.core.windows.net/path/data.csv`
- A database table or query
- A public URL (`https://...`)

```python
source = "s3://my-bucket/wine/wine_quality_cleaned_dropna.csv"
df = pd.read_csv(source)  # needs the matching filesystem package, e.g. s3fs
dataset = mlflow.data.from_pandas(
    df, source=source, name="wine_quality_cleaned_dropna", targets="quality"
)
```

When you do this, also add `data/<folder>/*` and `!data/<folder>/.gitkeep` rules back to the
root `.gitignore` for each data folder, so the local copies are not committed.
