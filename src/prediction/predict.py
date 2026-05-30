from pathlib import Path
import joblib
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ALLOWED_TYPES = ["H", "L", "M"]


MODEL_PATHS = [
    PROJECT_ROOT / "models" / "final_baseline_random_forest.joblib",
    PROJECT_ROOT / "models" / "random_forest_baseline.joblib",
]

PREPROCESSOR_PATHS = [
    PROJECT_ROOT / "models" / "preprocessor.joblib",
    PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib",
]


def find_existing_path(paths, artifact_name):
    for path in paths:
        if path.exists():
            return path

    paths_text = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(
        f"Could not find {artifact_name}. Checked these paths:\n{paths_text}"
    )


def load_artifacts():
    model_path = find_existing_path(MODEL_PATHS, "trained model")
    preprocessor_path = find_existing_path(PREPROCESSOR_PATHS, "preprocessor")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def validate_input(input_data):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("Input must be a dictionary or a pandas DataFrame.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    extra_columns = [col for col in df.columns if col not in REQUIRED_COLUMNS]

    df = df[REQUIRED_COLUMNS].copy()

    df["Type"] = df["Type"].astype(str)

    invalid_types = sorted(set(df["Type"]) - set(ALLOWED_TYPES))
    if invalid_types:
        raise ValueError(
            f"Invalid machine Type values: {invalid_types}. "
            f"Allowed values are: {ALLOWED_TYPES}"
        )

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df, extra_columns


def predict_machine_failure(input_data, threshold=0.50):
    model, preprocessor = load_artifacts()

    clean_input, extra_columns = validate_input(input_data)

    processed_input = preprocessor.transform(clean_input)

    failure_probability = model.predict_proba(processed_input)[:, 1]

    predicted_failure = (failure_probability >= threshold).astype(int)

    result = clean_input.copy()
    result["failure_probability"] = failure_probability
    result["failure_probability_percent"] = np.round(failure_probability * 100, 2)
    result["predicted_failure"] = predicted_failure
    result["prediction_label"] = np.where(
        predicted_failure == 1,
        "Failure risk",
        "Normal"
    )
    result["threshold_used"] = threshold

    return result, extra_columns


if __name__ == "__main__":
    sample_input = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }

    prediction, ignored_columns = predict_machine_failure(sample_input)
    print(prediction)