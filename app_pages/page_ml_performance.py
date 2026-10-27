"""ML Performance page: dataset split, learning curves and test results."""
import plotly.express as px
import streamlit as st

from src.data_management import file_exists, load_csv_file, output_file
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_body():
    st.header("ML Performance Metrics")

    # ---- Overall verdict first, so the answer is easy to find ----------------
    if file_exists("evaluation.pkl"):
        table, evaluation = load_test_evaluation()
        if evaluation["accuracy"] >= evaluation["target_accuracy"]:
            st.success(
                f"**The model meets the business requirement.** It reached "
                f"**{evaluation['accuracy']:.2%}** accuracy on unseen test images, "
                f"above the agreed target of {evaluation['target_accuracy']:.0%}."
            )
        else:
            st.error(
                f"**The model does not yet meet the business requirement.** It reached "
                f"{evaluation['accuracy']:.2%} accuracy on unseen test images, below the "
                f"agreed target of {evaluation['target_accuracy']:.0%}."
            )
    else:
        st.warning("Run notebook 03 to create the evaluation results.")
        return

    # ---- Dataset split (interactive) -----------------------------------------
    st.subheader("Images per set and label")
    if file_exists("labels_distribution.csv"):
        df_freq = load_csv_file("labels_distribution.csv")
        fig = px.bar(df_freq, x="Set", y="Frequency", color="Label", barmode="group",
                     color_discrete_map={"healthy": "#2e7d32",
                                         "powdery_mildew": "#9e9e9e"})
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            "**Interpretation:** the data is split 70% train, 10% validation and 20% "
            "test, and both labels are equally represented in each set. Balanced "
            "classes mean accuracy is a fair measure of performance. Hover over the "
            "bars to see exact numbers."
        )

    # ---- Model comparison ----------------------------------------------------
    if file_exists("model_comparison.csv"):
        st.subheader("Model versions compared")
        st.dataframe(load_csv_file("model_comparison.csv"), hide_index=True)
        st.write(
            "**Interpretation:** two versions were trained. Version A had no image "
            "augmentation and light dropout; version B added augmentation and stronger "
            "dropout to reduce overfitting. The version with the lowest validation "
            f"loss (**{evaluation['model_version']}**) was kept as the final model."
        )

    # ---- Learning curves -----------------------------------------------------
    st.subheader("Model training history")
    col1, col2 = st.columns(2)
    if file_exists("model_training_accuracy.png"):
        col1.image(output_file("model_training_accuracy.png"),
                   caption="Training and validation accuracy per epoch")
    if file_exists("model_training_loss.png"):
        col2.image(output_file("model_training_loss.png"),
                   caption="Training and validation loss per epoch")
    st.write(
        "**Interpretation:** accuracy rises and loss falls for both the training and "
        "validation sets, and the two lines stay close together. This shows the model "
        "learned general patterns rather than memorising the training images "
        "(no strong overfitting). Early stopping ended training once the validation "
        "loss stopped improving."
    )

    # ---- Test set ------------------------------------------------------------
    st.subheader("Performance on the test set")
    st.dataframe(table, hide_index=True)
    if file_exists("confusion_matrix.png"):
        st.image(output_file("confusion_matrix.png"),
                 caption="Confusion matrix on the test set", width=450)
    st.write(
        "**Interpretation:** the confusion matrix compares the actual label (rows) "
        "with the model's prediction (columns). The diagonal shows correct "
        "predictions; any numbers off the diagonal are leaves the model got wrong."
    )
