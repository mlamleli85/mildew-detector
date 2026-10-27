"""Project Hypotheses page: what we expected and how each was checked."""
import streamlit as st

from src.data_management import (
    file_exists,
    load_json_file,
    load_pkl_file,
    output_file,
)


def show_hypothesis_2_result():
    """Show the brightness test numbers, box plot and conclusion."""
    result = load_json_file("brightness_test.json")
    col1, col2, col3 = st.columns(3)
    col1.metric("Healthy - mean brightness",
                f"{result['healthy_mean_brightness']:.3f}")
    col2.metric("Mildew - mean brightness",
                f"{result['mildew_mean_brightness']:.3f}")
    col3.metric("p-value", f"{result['p_value']:.2e}")

    if file_exists("brightness_boxplot.png"):
        st.image(output_file("brightness_boxplot.png"),
                 caption="Mean image brightness per label", width=500)

    brighter = (result["mildew_mean_brightness"]
                > result["healthy_mean_brightness"])

    if result["significant"] and brighter:
        st.success(
            "**Conclusion:** confirmed. Infected leaves are significantly "
            "brighter (p < 0.05). The box plot shows the mildew group sits "
            "higher, although the two groups overlap, so brightness alone "
            "is not enough to classify a single leaf."
        )
    elif result["significant"]:
        st.warning(
            "**Conclusion:** rejected. There is a significant difference "
            "in brightness (p < 0.05), but in the opposite direction: "
            "healthy leaves are brighter on average than infected leaves. "
            "The mildew coating is a thin, greyish film that dulls the "
            "leaf rather than making it brighter. Lighting or background "
            "differences between the photos may also play a part, so the "
            "model should be tested on photos taken in normal field "
            "conditions."
        )
    else:
        st.warning(
            "**Conclusion:** not confirmed. There is no significant "
            "difference in brightness, so the model needs to learn shapes "
            "and textures rather than overall colour."
        )


def page_project_hypothesis_body():
    st.header("Project Hypotheses and Validation")

    # ---- Hypothesis 1 --------------------------------------------------
    st.subheader("Hypothesis 1: infected leaves look visibly different")
    st.info(
        "**Hypothesis:** leaves with powdery mildew show white, powdery "
        "marks that set them apart from healthy leaves.\n\n"
        "**Validation:** image montage, average images and difference "
        "image (see the Leaf Visualiser page)."
    )
    st.success(
        "**Conclusion:** confirmed. The montages clearly show white "
        "patches on infected leaves, and the average images of the two "
        "labels look different."
    )

    # ---- Hypothesis 2 --------------------------------------------------
    st.subheader("Hypothesis 2: infected leaves are brighter on average")
    st.info(
        "**Hypothesis:** because the mildew is white, infected leaves have "
        "a higher mean pixel brightness than healthy leaves.\n\n"
        "**Validation:** mean brightness of each image, compared with a "
        "Mann-Whitney U test (significance level 0.05)."
    )
    if file_exists("brightness_test.json"):
        show_hypothesis_2_result()
    else:
        st.warning("Run notebook 02 to create the brightness test results.")

    # ---- Hypothesis 3 --------------------------------------------------
    st.subheader(
        "Hypothesis 3: a CNN can detect mildew with at least 97% accuracy"
    )
    st.info(
        "**Hypothesis:** a convolutional neural network trained on the "
        "leaf images can classify unseen leaves with at least 97% "
        "accuracy.\n\n"
        "**Validation:** accuracy on the test set, which the model never "
        "saw during training (see the ML Performance page)."
    )
    if file_exists("evaluation.pkl"):
        evaluation = load_pkl_file("evaluation.pkl")
        accuracy = evaluation["accuracy"]
        if accuracy >= evaluation["target_accuracy"]:
            st.success(f"**Conclusion:** confirmed. Test accuracy is "
                       f"{accuracy:.2%}.")
        else:
            st.warning(f"**Conclusion:** not confirmed. Test accuracy is "
                       f"{accuracy:.2%}.")

    st.subheader("Recommended next steps for the client")
    st.write(
        "* Roll the detector out to farm staff: one photo per tree "
        "instead of a 30-minute manual inspection.\n"
        "* Keep a manual check for low-confidence predictions "
        "(below 80%).\n"
        "* Collect photos from other crops to extend the system to other "
        "pests and diseases."
    )
