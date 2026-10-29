"""ML Performance page: dataset split, learning curves and test results."""
import plotly.express as px
import streamlit as st

from src.data_management import file_exists, load_csv_file, output_file
from src.machine_learning.evaluate_clf import load_test_evaluation

LABEL_COLOURS = {"healthy": "#2e7d32", "powdery_mildew": "#9e9e9e"}


def show_verdict(evaluation):
    """State clearly whether the model met the business target."""
    accuracy = evaluation["accuracy"]
    target = evaluation["target_accuracy"]
    if accuracy >= target:
        st.success(
            f"**The model meets the business requirement.** It reached "
            f"**{accuracy:.2%}** accuracy on unseen test images, above "
            f"the agreed target of {target:.0%}."
        )
    else:
        st.error(
            f"**The model does not yet meet the business requirement.** "
            f"It reached {accuracy:.2%} accuracy on unseen test images, "
            f"below the agreed target of {target:.0%}."
        )


def show_label_distribution():
    """Interactive bar chart of images per set and label."""
    st.subheader("Images per set and label")
    df_freq = load_csv_file("labels_distribution.csv")
    fig = px.bar(df_freq, x="Set", y="Frequency", color="Label",
                 barmode="group", color_discrete_map=LABEL_COLOURS)
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        "**Interpretation:** the data is split 70% train, 10% validation "
        "and 20% test, and both labels are equally represented in each "
        "set. Balanced classes mean accuracy is a fair measure of "
        "performance. Hover over the bars to see exact numbers."
    )


def show_learning_curves():
    """Training and validation accuracy and loss per epoch."""
    st.subheader("Model training history")
    col1, col2 = st.columns(2)
    if file_exists("model_training_accuracy.png"):
        col1.image(output_file("model_training_accuracy.png"),
                   caption="Training and validation accuracy per epoch")
    if file_exists("model_training_loss.png"):
        col2.image(output_file("model_training_loss.png"),
                   caption="Training and validation loss per epoch")
    st.write(
        "**Interpretation:** the validation loss was lowest after the "
        "first epoch. After that, the training loss kept falling while "
        "the validation loss rose, an early sign of overfitting (the "
        "model starting to memorise the training images). Early "
        "stopping halted training and restored the weights from the "
        "best epoch, so the saved model is the one that generalised "
        "best."
    )


def page_ml_performance_body():
    st.header("ML Performance Metrics")

    # Overall verdict first, so the answer is easy to find
    if not file_exists("evaluation.pkl"):
        st.warning("Run notebook 03 to create the evaluation results.")
        return
    table, evaluation = load_test_evaluation()
    show_verdict(evaluation)

    if file_exists("labels_distribution.csv"):
        show_label_distribution()

    if file_exists("model_comparison.csv"):
        st.subheader("Model versions compared")
        st.dataframe(load_csv_file("model_comparison.csv"), hide_index=True)
        st.write(
            "**Interpretation:** two versions were trained. Version A had "
            "no image augmentation and light dropout; version B added "
            "augmentation and stronger dropout to reduce overfitting. The "
            "version with the lowest validation loss "
            f"(**{evaluation['model_version']}**) was kept as the final "
            "model."
        )

    show_learning_curves()

    st.subheader("Performance on the test set")
    st.dataframe(table, hide_index=True)
    if file_exists("confusion_matrix.png"):
        st.image(output_file("confusion_matrix.png"),
                 caption="Confusion matrix on the test set", width=450)
    st.write(
        "**Interpretation:** the confusion matrix compares the actual "
        "label (rows) with the model's prediction (columns). 843 of 844 "
        "test leaves were classified correctly. Every infected leaf was "
        "detected; one healthy leaf was wrongly flagged as powdery "
        "mildew. This is the safer error for the client: no infected "
        "tree is missed."
    )
