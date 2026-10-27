"""Leaf Visualiser page: answers business requirement 1."""
import streamlit as st

from src.data_management import file_exists, output_file


def show_output_image(file_name, caption):
    """Show a saved plot, or a friendly message if the notebook has not been run."""
    if file_exists(file_name):
        st.image(output_file(file_name), caption=caption)
    else:
        st.warning(f"`{file_name}` not found. Run notebook 02 to create it.")


def page_leaves_visualizer_body():
    st.header("Leaf Visualiser")
    st.info(
        "**Business requirement 1:** the client wants a study that visually "
        "differentiates a healthy cherry leaf from one with powdery mildew."
    )

    if st.checkbox("Average and variability images"):
        show_output_image("avg_var_healthy.png",
                          "Average and variability image - healthy leaves")
        show_output_image("avg_var_powdery_mildew.png",
                          "Average and variability image - leaves with powdery mildew")
        st.warning(
            "**Interpretation:** the average healthy leaf is an even, darker green. "
            "The average infected leaf looks lighter and paler, matching the white "
            "powdery patches. The variability images show where leaves differ most "
            "from each other: mostly along the edges and leaf shape. Averages blur "
            "the individual patches, so they hint at the difference rather than "
            "showing it clearly - which is why we also train a model."
        )

    if st.checkbox("Difference between average healthy and average infected leaf"):
        show_output_image("avg_diff.png",
                          "Average healthy leaf, average infected leaf and the difference")
        st.warning(
            "**Interpretation:** brighter areas in the difference image are where the "
            "two average leaves differ most. The difference is spread across the leaf "
            "surface rather than in one fixed place, because mildew patches appear in "
            "different spots on each leaf."
        )

    if st.checkbox("Image montage"):
        st.write("Choose a label to see a grid of real leaf photos from the training set.")
        label = st.selectbox("Label", ["healthy", "powdery_mildew"],
                             format_func=lambda x: x.replace("_", " ").capitalize())
        show_output_image(f"montage_{label}.png", f"Montage of {label.replace('_', ' ')} leaves")
        st.warning(
            "**Interpretation:** healthy leaves are smooth and evenly green. Leaves "
            "with powdery mildew show clear white or grey powdery spots and streaks, "
            "often across large parts of the leaf. This is the visual sign the model "
            "learns to detect."
        )
