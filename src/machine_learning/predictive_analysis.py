"""Load the trained model, prepare uploaded images and render predictions."""
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from src.data_management import load_pkl_file, output_file

MODEL_FILE = "mildew_detector_model.keras"


@st.cache_resource
def load_mildew_model():
    """Load the model once and keep it in memory between page reloads."""
    return load_model(output_file(MODEL_FILE))


def resize_input_image(img):
    """Resize an uploaded PIL image to the shape the model was trained on.

    Pixel scaling (0-255 -> 0-1) happens inside the model's Rescaling layer.
    """
    image_shape = load_pkl_file("image_shape.pkl")
    img_resized = img.convert("RGB").resize(image_shape[:2], Image.LANCZOS)
    return np.expand_dims(np.array(img_resized, dtype="float32"), axis=0)


def predict_probability(image_array):
    """Return (predicted label, probability of that label)."""
    class_names = load_pkl_file("class_names.pkl")  # ['healthy', 'powdery_mildew']
    model = load_mildew_model()
    mildew_probability = float(model.predict(image_array, verbose=0)[0][0])

    if mildew_probability >= 0.5:
        return class_names[1], mildew_probability
    return class_names[0], 1 - mildew_probability


def plot_prediction_probabilities(label, probability):
    """Interactive bar chart of the probability for each class."""
    class_names = load_pkl_file("class_names.pkl")
    other = [c for c in class_names if c != label][0]
    fig = px.bar(
        x=[label, other], y=[probability, 1 - probability],
        labels={"x": "Diagnosis", "y": "Probability"},
        color=[label, other],
        color_discrete_map={"healthy": "#2e7d32", "powdery_mildew": "#b0b0b0"},
        range_y=[0, 1], height=320,
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
