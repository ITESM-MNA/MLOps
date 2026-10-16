import pandas as pd

from mlops.dataset import clean_data
from mlops.features import build_features
from mlops.transformers import OutlierClipper


def test_clean_data_uses_snake_case_and_drops_duplicates():
    df = pd.DataFrame({"fixed acidity": [7.0, 7.0, 8.0], "pH": [3.0, 3.0, None]})
    out = clean_data(df)
    assert list(out.columns) == ["fixed_acidity", "ph"]
    assert len(out) == 1


def test_build_features_keeps_rows():
    df = pd.DataFrame({"fixed_acidity": [7.0, 8.0], "volatile_acidity": [0.5, 0.25]})
    out = build_features(df)
    assert len(out) == len(df)
    assert out["total_acidity"].tolist() == [7.5, 8.25]


def test_outlier_clipper_caps_values():
    X = pd.DataFrame({"a": range(100), "b": [1.0] * 99 + [1000.0]})
    clipper = OutlierClipper(q=0.9).fit(X)
    assert list(clipper.upper_.index) == ["a", "b"]

    out = clipper.transform(X)
    assert (out <= clipper.upper_).all().all()
    assert out.shape == X.shape
