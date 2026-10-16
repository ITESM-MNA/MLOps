"""Build data/processed/wine_quality_train.csv and wine_quality_test.csv.

Starts from the cleaned white-wine dataset produced by notebook 1.4.

    python scripts/prepare_data.py
"""

from sklearn.model_selection import train_test_split

from mlops.config import RANDOM_STATE, SOURCE_DATA_FILE, TEST_FILE, TRAIN_FILE
from mlops.dataset import clean_data, load_data
from mlops.features import build_features

df = build_features(clean_data(load_data(SOURCE_DATA_FILE)))
train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)

print(f"{TRAIN_FILE.name}: {len(train)} rows, {TEST_FILE.name}: {len(test)} rows")
