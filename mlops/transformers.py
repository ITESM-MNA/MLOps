"""Custom scikit-learn transformers."""

from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip each column to the quantile `q` learned in fit."""

    def __init__(self, q=0.99):
        self.q = q

    def fit(self, X, y=None):
        self.upper_ = X.quantile(self.q)
        return self

    def transform(self, X):
        return X.clip(upper=self.upper_, axis=1)
