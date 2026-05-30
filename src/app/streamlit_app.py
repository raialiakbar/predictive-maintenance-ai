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


st.title("⚙️ Predictive Maintenance AI")
st.write(
    "This app predicts whether a machine is at risk of failure using the trained "
    "Random Forest baseline model."
)


st.sidebar.header("Machine Input")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    options=["L", "M", "H"],
    index=0
)

air_temperature = st.sidebar.number_input(
    "Air temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.number_input(
    "Process temperature [K]",
    min_value=250.0,
    max_value=450.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational speed [rpm]",
    min_value=0,
    max_value=5000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool wear [min]",
    min_value=0,
    max_value=400,
    value=100,
    step=1
)

threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)


input_data = {
    "Type": machine_type,
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
}


st.subheader("Single Machine Prediction")

input_df = pd.DataFrame([input_data])
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Failure Risk"):
    try:
        prediction, ignored_columns = predict_machine_failure(
            input_data,
            threshold=threshold
        )

        probability = prediction.loc[0, "failure_probability"]
        probability_percent = prediction.loc[0, "failure_probability_percent"]
        label = prediction.loc[0, "prediction_label"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Failure Probability", f"{probability_percent}%")
        col2.metric("Prediction", label)
        col3.metric("Threshold", threshold)

        if probability >= threshold:
            st.error("⚠️ The model predicts a machine failure risk.")
        else:
            st.success("✅ The model predicts normal machine condition.")

        st.subheader("Prediction Details")
        st.dataframe(prediction, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()

st.subheader("Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file with the required raw input columns",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Uploaded data preview:")
        st.dataframe(batch_df.head(), use_container_width=True)

        batch_predictions, ignored_columns = predict_machine_failure(
            batch_df,
            threshold=threshold
        )

        if ignored_columns:
            st.warning(
                f"The following extra columns were ignored: {ignored_columns}"
            )

        st.write("Prediction results:")
        st.dataframe(batch_predictions, use_container_width=True)

        csv_output = batch_predictions.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions as CSV",
            data=csv_output,
            file_name="machine_failure_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Batch prediction failed: {error}")


st.divider()

st.caption(
    "Model: Random Forest baseline | Input: raw machine sensor values | "
    "Target: Machine failure"
)