"""Leaf Visualiser page: answers business requirement 1."""
import streamlit as st

from src.data_management import file_exists, output_file


def show_output_image(file_name, caption):
    """Show a saved plot, or a message if the notebook has not been run."""
    if file_exists(file_name):
        st.image(output_file(file_name), caption=caption)
    else:
        st.warning(f"`{file_name}` not found. Run notebook 02 to create it.")


def readable(label):
    """Turn 'powdery_mildew' into 'Powdery mildew' for display."""
    return label.replace("_", " ").capitalize()


def page_leaves_visualizer_body():
    st.header("Leaf Visualiser")
    st.info(
        "**Business requirement 1:** the client wants a study that "
        "visually differentiates a healthy cherry leaf from one with "
        "powdery mildew."
    )

    if st.checkbox("Average and variability images"):
        show_output_image(
            "avg_var_healthy.png",
            "Average and variability image - healthy leaves")
        show_output_image(
            "avg_var_powdery_mildew.png",
            "Average and variability image - leaves with powdery mildew")
        st.warning(
            "**Interpretation:** the average healthy leaf is a saturated "
            "green on a light background; the average infected leaf is a "
            "duller, greyer green on a darker background. The variability "
            "images show where leaves differ most from each other "
            "(lighter areas = more variation). Mildew patches appear in "
            "different places on each leaf, so averaging blurs them; "
            "this is why we also train a model."
        )

    if st.checkbox("Difference between average healthy and infected leaf"):
        show_output_image(
            "avg_diff.png",
            "Average healthy leaf, average infected leaf and the difference")
        st.warning(
            "**Interpretation:** darker areas are where the two average "
            "images are similar; brighter areas are where they differ. "
            "The centre, where the leaf sits, is dark, while the edges "
            "are bright: the biggest difference between the averages is "
            "the background, which is lighter in the healthy photos. "
            "This suggests the two groups were photographed in slightly "
            "different conditions, so the model should be checked on "
            "field photos to make sure it detects the disease and not "
            "the background."
        )

    if st.checkbox("Image montage"):
        st.write("Choose a label to see a grid of real leaf photos from "
                 "the training set.")
        label = st.selectbox("Label", ["healthy", "powdery_mildew"],
                             format_func=readable)
        show_output_image(f"montage_{label}.png",
                          f"Montage of {readable(label).lower()} leaves")
        st.warning(
            "**Interpretation:** healthy leaves are smooth and evenly "
            "green. Leaves with powdery mildew show white or grey powdery "
            "spots and streaks, often across large parts of the leaf. "
            "This is the visual sign the model learns to detect."
        )
