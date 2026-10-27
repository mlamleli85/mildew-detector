"""Helpers for loading saved project outputs and preparing downloads."""
import os
import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

VERSION = "v1"
OUTPUT_PATH = os.path.join("outputs", VERSION)


def output_file(file_name):
    """Return the path of a file in the current outputs version folder."""
    return os.path.join(OUTPUT_PATH, file_name)


def file_exists(file_name):
    return os.path.isfile(output_file(file_name))


@st.cache_data
def load_pkl_file(file_name):
    """Load a pickle file saved by the notebooks (e.g. evaluation.pkl)."""
    return joblib.load(output_file(file_name))


@st.cache_data
def load_csv_file(file_name):
    return pd.read_csv(output_file(file_name))


@st.cache_data
def load_json_file(file_name):
    with open(output_file(file_name), encoding="utf-8") as f:
        return json.load(f)


def report_file_name():
    """Timestamped name for the downloadable prediction report."""
    return f"mildew_report_{datetime.now():%Y-%m-%d_%H-%M}.csv"
