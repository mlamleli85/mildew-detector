"""Project Summary page: the business problem, dataset and requirements."""
import streamlit as st

DATASET_URL = "https://www.kaggle.com/codeinstitute/cherry-leaves"
README_URL = "https://github.com/mlamleli85/mildew-detector#readme"


def page_summary_body():
    st.header("Project Summary")

    st.info(
        "**The problem**\n\n"
        "Farmy & Foods grows cherries on thousands of trees across several "
        "farms. Some trees are affected by **powdery mildew**, a fungal "
        "disease that shows as white, powdery patches on the leaves.\n\n"
        "Today an employee inspects each tree by hand, which takes about "
        "**30 minutes per tree**. This does not scale. This dashboard uses "
        "machine learning to tell from a single leaf photo whether a tree "
        "is healthy or infected, in seconds."
    )

    st.subheader("The dataset")
    st.write(
        "* About 4,200 photos of cherry leaves provided by Farmy & Foods, "
        "taken on their farms.\n"
        "* Two labels of roughly equal size: **healthy** and "
        "**powdery_mildew**.\n"
        f"* Source: [Kaggle - Cherry Leaves dataset]({DATASET_URL})."
    )

    st.subheader("Business requirements")
    st.success(
        "1. The client wants a study that **visually differentiates** a "
        "healthy cherry leaf from one with powdery mildew. "
        "*(See the **Leaf Visualiser** page.)*\n"
        "2. The client wants to **predict** whether a cherry leaf is "
        "healthy or has powdery mildew. "
        "*(See the **Mildew Detector** page.)*"
    )

    st.subheader("How to use this dashboard")
    st.write(
        "Use the menu on the left:\n\n"
        "* **Leaf Visualiser** - what healthy and infected leaves look "
        "like.\n"
        "* **Mildew Detector** - upload leaf photos and get a diagnosis.\n"
        "* **Project Hypotheses** - what we expected and how we checked "
        "it.\n"
        "* **ML Performance** - how well the model performs on unseen "
        "images."
    )

    st.write(f"For more details, read the project [README]({README_URL}).")
