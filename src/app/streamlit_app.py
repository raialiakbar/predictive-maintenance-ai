from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.prediction.predict import predict_machine_failure


st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


REQUIRED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


FEATURE_IMPORTANCE_PATHS = [
    PROJECT_ROOT / "reports" / "figures" / "random_forest_feature_importance.png",
    PROJECT_ROOT / "reports" / "random_forest_feature_importance.png",
]

CONFUSION_MATRIX_PATHS = [
    PROJECT_ROOT / "reports" / "figures" / "random_forest_confusion_matrix.png",
    PROJECT_ROOT / "reports" / "random_forest_confusion_matrix.png",
]

ROC_CURVE_PATHS = [
    PROJECT_ROOT / "reports" / "figures" / "random_forest_roc_curve.png",
    PROJECT_ROOT / "reports" / "random_forest_roc_curve.png",
]

PR_CURVE_PATHS = [
    PROJECT_ROOT / "reports" / "figures" / "random_forest_precision_recall_curve.png",
    PROJECT_ROOT / "reports" / "random_forest_precision_recall_curve.png",
]


def find_image_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def get_risk_level(probability):
    if probability < 0.30:
        return "Low risk", "The machine condition looks normal based on the model output."
    elif probability < 0.60:
        return "Medium risk", "The model sees some warning signs. The machine should be monitored."
    else:
        return "High risk", "The model predicts a high failure risk. Maintenance inspection is recommended."


def show_prediction_result(prediction):
    probability = prediction.loc[0, "failure_probability"]
    probability_percent = prediction.loc[0, "failure_probability_percent"]
    label = prediction.loc[0, "prediction_label"]
    threshold = prediction.loc[0, "threshold_used"]

    risk_level, risk_message = get_risk_level(probability)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Failure Probability", f"{probability_percent}%")
    col2.metric("Prediction", label)
    col3.metric("Risk Level", risk_level)
    col4.metric("Threshold", threshold)

    if probability >= threshold:
        st.error(f"⚠️ {risk_message}")
    else:
        st.success(f"✅ {risk_message}")

    st.subheader("Prediction Details")
    st.dataframe(prediction, use_container_width=True)


def create_sample_csv():
    sample_data = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
            },
            {
                "Type": "M",
                "Air temperature [K]": 302.5,
                "Process temperature [K]": 311.8,
                "Rotational speed [rpm]": 1350,
                "Torque [Nm]": 52.0,
                "Tool wear [min]": 180,
            },
            {
                "Type": "H",
                "Air temperature [K]": 298.4,
                "Process temperature [K]": 308.9,
                "Rotational speed [rpm]": 1700,
                "Torque [Nm]": 28.0,
                "Tool wear [min]": 60,
            },
        ]
    )

    return sample_data.to_csv(index=False).encode("utf-8")


st.title("⚙️ Predictive Maintenance AI")
st.write(
    "This app predicts machine failure risk using a trained Random Forest model. "
    "It accepts raw machine sensor values, applies the saved preprocessing pipeline, "
    "and returns a failure probability and prediction label."
)

with st.expander("Project Overview", expanded=True):
    st.write(
        """
        This project uses the AI4I 2020 Predictive Maintenance dataset for binary machine failure prediction.
        The goal is to predict whether a machine is likely to fail based on machine type, temperature,
        rotational speed, torque, and tool wear.
        """
    )

    st.write(
        """
        The app uses the same saved preprocessor that was fitted during training. This is important because
        the model expects the input features in the same processed format used during model training.
        """
    )

    st.code(
        """
Raw machine input
        ↓
Saved preprocessor
        ↓
Trained Random Forest model
        ↓
Failure probability + prediction label
        """,
        language="text"
    )


tab_single, tab_batch, tab_model = st.tabs(
    ["Single Prediction", "Batch Prediction", "Model Information"]
)


with tab_single:
    st.header("Single Machine Prediction")
    st.write("Enter raw machine values manually and predict the failure risk.")

    with st.sidebar:
        st.header("Machine Input")

        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            index=0,
            help="L = Low quality/product variant, M = Medium, H = High"
        )

        air_temperature = st.number_input(
            "Air temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process temperature [K]",
            min_value=250.0,
            max_value=450.0,
            value=310.0,
            step=0.1
        )

        rotational_speed = st.number_input(
            "Rotational speed [rpm]",
            min_value=0,
            max_value=5000,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool wear [min]",
            min_value=0,
            max_value=400,
            value=100,
            step=1
        )

        threshold = st.slider(
            "Prediction threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="If the failure probability is above this value, the machine is classified as failure risk."
        )

    input_data = {
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    input_df = pd.DataFrame([input_data])

    st.subheader("Current Input")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Failure Risk", type="primary"):
        try:
            prediction, ignored_columns = predict_machine_failure(
                input_data,
                threshold=threshold
            )

            show_prediction_result(prediction)

        except Exception as error:
            st.error(f"Prediction failed: {error}")


with tab_batch:
    st.header("Batch Prediction from CSV")
    st.write(
        "Upload a CSV file with raw machine data. The app will predict failure risk for each row."
    )

    st.subheader("Required CSV Columns")
    st.code("\n".join(REQUIRED_COLUMNS), language="text")

    st.info(
        "The uploaded CSV must contain the required raw input columns. "
        "Extra columns such as UDI, Product ID, Machine failure, TWF, HDF, PWF, OSF, or RNF are ignored."
    )

    st.download_button(
        label="Download Sample Input CSV",
        data=create_sample_csv(),
        file_name="sample_machine_input.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    batch_threshold = st.slider(
        "Batch prediction threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            batch_predictions, ignored_columns = predict_machine_failure(
                batch_df,
                threshold=batch_threshold
            )

            if ignored_columns:
                st.warning(
                    f"The following extra columns were ignored during prediction: {ignored_columns}"
                )

            st.subheader("Batch Prediction Results")
            st.dataframe(batch_predictions, use_container_width=True)

            total_rows = len(batch_predictions)
            predicted_failures = int(batch_predictions["predicted_failure"].sum())
            predicted_normal = total_rows - predicted_failures
            average_probability = batch_predictions["failure_probability"].mean() * 100

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Machines", total_rows)
            col2.metric("Predicted Normal", predicted_normal)
            col3.metric("Predicted Failure Risk", predicted_failures)
            col4.metric("Average Failure Probability", f"{average_probability:.2f}%")

            csv_output = batch_predictions.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="machine_failure_predictions.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Batch prediction failed: {error}")


with tab_model:
    st.header("Model Information")

    st.subheader("Selected Model")
    st.write(
        "The app uses a Random Forest classifier as the selected baseline model. "
        "This model was chosen because it gave the best overall balance between detecting failures "
        "and avoiding too many false alarms."
    )

    metrics_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC",
                "PR-AUC",
            ],
            "Score": [
                0.971,
                0.556,
                0.735,
                0.633,
                0.964,
                0.624,
            ],
        }
    )

    st.subheader("Test Performance")
    st.dataframe(metrics_data, use_container_width=True)

    st.write(
        "The default threshold is 0.50. A threshold of 0.45 was also tested, but the default threshold "
        "was kept because it gave a better overall test F1-score and a more balanced result."
    )

    st.subheader("Feature Importance")

    feature_importance_path = find_image_path(FEATURE_IMPORTANCE_PATHS)

    if feature_importance_path is not None:
        st.image(str(feature_importance_path), caption="Random Forest Feature Importance")
    else:
        st.warning("Feature importance image was not found.")

    st.write(
        "The most important features for the Random Forest model were rotational speed, torque, "
        "and tool wear. Product type had lower importance compared to the numerical sensor features."
    )

    st.subheader("Evaluation Figures")

    col1, col2 = st.columns(2)

    confusion_matrix_path = find_image_path(CONFUSION_MATRIX_PATHS)
    roc_curve_path = find_image_path(ROC_CURVE_PATHS)
    pr_curve_path = find_image_path(PR_CURVE_PATHS)

    with col1:
        if confusion_matrix_path is not None:
            st.image(str(confusion_matrix_path), caption="Confusion Matrix")
        else:
            st.warning("Confusion matrix image was not found.")

    with col2:
        if roc_curve_path is not None:
            st.image(str(roc_curve_path), caption="ROC Curve")
        else:
            st.warning("ROC curve image was not found.")

    if pr_curve_path is not None:
        st.image(str(pr_curve_path), caption="Precision-Recall Curve")
    else:
        st.warning("Precision-recall curve image was not found.")


st.divider()

st.caption(
    "Predictive Maintenance AI | Random Forest baseline model | "
    "Raw input → saved preprocessor → failure prediction"
)