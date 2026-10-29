"""Helpers to show the saved model evaluation on the dashboard."""
import pandas as pd

from src.data_management import load_pkl_file


def load_test_evaluation():
    """Return the test-set evaluation as a small table and the raw dict."""
    evaluation = load_pkl_file("evaluation.pkl")
    # Both values as text so the column has one data type
    results = [f"{evaluation['loss']:.4f}", f"{evaluation['accuracy']:.2%}"]
    table = pd.DataFrame({"Metric": ["Loss", "Accuracy"],
                          "Test set result": results})
    return table, evaluation
