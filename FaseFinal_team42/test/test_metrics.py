import pytest
from src.evaluate_model import summary_basic, summary_best_f1

def test_summary_basic():
    """Test basic summary metrics calculation."""
    y_true = [0, 1, 1, 0, 1]
    y_score = [0.1, 0.9, 0.8, 0.2, 0.7]
    metrics = summary_basic("test", y_true, y_score)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

def test_summary_best_f1():
    """Test best F1 score calculation."""
    y_true = [0, 1, 1, 0, 1]
    y_score = [0.1, 0.9, 0.8, 0.2, 0.7]
    metrics = summary_best_f1("test", y_true, y_score)
    assert "best_f1" in metrics
    assert metrics["best_f1"] > 0
