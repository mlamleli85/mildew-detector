"""Helpers to show the saved model evaluation on the dashboard."""
import pandas as pd

from src.data_management import load_pkl_file


def load_test_evaluation():
    """Return the test-set evaluation as a small table and the raw dict."""
    evaluation = load_pkl_file("evaluation.pkl")
    table = pd.DataFrame(
        {"Metric": ["Loss", "Accuracy"],
         # Both values as text so the column has one data type
         "Test set result": [f"{evaluation['loss']:.4f}", f"{evaluation['accuracy']:.2%}"]}
    )
    return table, evaluation
