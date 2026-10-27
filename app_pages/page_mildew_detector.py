"""Mildew Detector page: answers business requirement 2 with live predictions."""
import pandas as pd
import streamlit as st
from PIL import Image

from src.data_management import file_exists, report_file_name
from src.machine_learning.predictive_analysis import (
    MODEL_FILE,
    plot_prediction_probabilities,
    predict_probability,
    resize_input_image,
)


def page_mildew_detector_body():
    st.header("Mildew Detector")
    st.info(
        "**Business requirement 2:** the client wants to predict whether a cherry "
        "leaf is healthy or has powdery mildew."
    )
    st.write(
        "Upload one or more photos of cherry leaves. For test images, download the "
        "[Kaggle cherry leaves dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)."
    )

    if not file_exists(MODEL_FILE):
        st.error("The trained model was not found. Run notebook 03 to create it.")
        return

    uploaded_images = st.file_uploader(
        "Upload cherry leaf images (PNG or JPG). You can select more than one.",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True,
    )

    if not uploaded_images:
        return

    results = []
    for uploaded in uploaded_images:
        st.divider()
        try:
            img = Image.open(uploaded)
        except Exception:
            st.error(f"`{uploaded.name}` could not be read as an image and was skipped.")
            continue

        col_image, col_result = st.columns([1, 2])
        with col_image:
            st.image(img, caption=f"{uploaded.name} ({img.size[0]} x {img.size[1]} px)")

        with col_result:
            with st.spinner("Analysing leaf..."):
                label, probability = predict_probability(resize_input_image(img))

            readable = label.replace("_", " ")
            if label == "healthy":
                st.success(f"**Healthy** - {probability:.1%} confidence")
            else:
                st.error(f"**Powdery mildew detected** - {probability:.1%} confidence. "
                         "Treat this tree with the anti-fungal compound.")
            plot_prediction_probabilities(label, probability)

        results.append({"Image": uploaded.name, "Prediction": readable,
                        "Confidence": round(probability, 4)})

    if results:
        st.divider()
        st.subheader("Analysis report")
        report = pd.DataFrame(results)
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button(
            "Download report as CSV", data=report.to_csv(index=False).encode("utf-8"),
            file_name=report_file_name(), mime="text/csv",
        )
