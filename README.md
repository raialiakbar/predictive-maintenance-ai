# Predictive Maintenance AI

This project predicts machine failure using the **AI4I 2020 Predictive Maintenance Dataset**. The goal is to build a complete machine learning workflow, starting from data understanding and preprocessing, then model training and evaluation, and finally a working prediction app.

The final application allows users to enter raw machine sensor values manually or upload a CSV file and receive machine failure predictions through a Streamlit web interface.

---

## Project Goal

The goal of this project is to predict whether a machine is likely to fail based on operational sensor data.

This is a binary classification problem:

```text
0 = No machine failure
1 = Machine failure
```

The project focuses on building a practical end-to-end machine learning pipeline that can be used outside of Jupyter notebooks.

---

## Dataset

The project uses the AI4I 2020 Predictive Maintenance dataset.

The raw dataset contains machine information, sensor values, and failure indicators. For the final prediction task, the target column is:

```text
Machine failure
```

The main input features used for prediction are:

```text
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

The following columns were removed during preprocessing because they are identifiers or leakage-related failure-type columns:

```text
UDI
Product ID
TWF
HDF
PWF
OSF
RNF
```

---

## Project Workflow

The project was developed in the following stages:

```text
1. Data understanding and exploratory data analysis
2. Data preprocessing
3. Baseline model training
4. Model evaluation
5. Prediction pipeline
6. Streamlit web app
```

---

## Preprocessing

The preprocessing pipeline was created using `ColumnTransformer`.

Numerical features were scaled using `StandardScaler`:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

The categorical feature was encoded using `OneHotEncoder`:

```text
Type
```

After preprocessing, the final processed features were:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Type_H
Type_L
Type_M
```

The fitted preprocessor was saved as:

```text
data/processed/preprocessor.joblib
```

This saved preprocessor is reused in the prediction app so that new input data is transformed in the same way as the training data.

---

## Models Trained

Several baseline models were trained and compared:

```text
Dummy Classifier
Logistic Regression
Decision Tree
Random Forest
```

Because the dataset is imbalanced, accuracy alone was not enough to evaluate the models. The main focus was on precision, recall, F1-score, ROC-AUC, and PR-AUC.

---

## Final Selected Model

The final selected model is:

```text
Random Forest Classifier
```

The Random Forest model was selected because it gave the best overall balance between detecting machine failures and avoiding too many false alarms.

The model was saved as:

```text
models/final_baseline_random_forest.joblib
```

---

## Model Performance

The selected Random Forest model achieved the following test performance:

```text
Accuracy:  0.971
Precision: 0.556
Recall:    0.735
F1-score:  0.633
ROC-AUC:   0.964
PR-AUC:    0.624
```

The model performed much better than the dummy classifier, which achieved high accuracy only because most machines in the dataset did not fail. However, the dummy model failed to detect actual machine failures.

The default prediction threshold used in the app is:

```text
0.50
```

A threshold of `0.45` was also tested, but the default threshold was kept because it gave a better overall test F1-score and a more balanced result.

---

## Feature Importance

The Random Forest model showed that the most important features were mainly numerical sensor features.

The most important features were:

```text
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

Temperature features also contributed to the prediction, while product type had lower importance compared to the numerical sensor values.

---

## Streamlit Prediction App

The project includes a Streamlit web app for making machine failure predictions.

The app supports:

```text
Single machine prediction using manual input values
Batch prediction from uploaded CSV files
Automatic preprocessing using the saved preprocessor
Failure probability output
Prediction label output
Adjustable prediction threshold
Risk-level message
Sample CSV download
CSV download for batch prediction results
Model information tab with evaluation results
Feature importance and evaluation figures
```

---

## How the App Works

The app follows this prediction workflow:

```text
Raw machine input
        ↓
Saved preprocessing pipeline
        ↓
Trained Random Forest model
        ↓
Failure probability + prediction label
```

The user provides raw machine data. The app then applies the saved preprocessor and passes the processed data to the trained Random Forest model.

The output contains:

```text
Failure probability
Failure probability percentage
Predicted class
Prediction label
Threshold used
Risk level
```

---

## Required App Input

The app expects raw input data with the following columns:

```text
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

The `Type` column should contain one of these values:

```text
L
M
H
```

For CSV upload, extra columns are allowed. Columns such as `UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are ignored during prediction.

Important: the app expects raw input data. The already processed files should not be uploaded into the app:

```text
data/processed/train_processed.csv
data/processed/test_processed.csv
```

---

## Running the App Locally

First, install the required packages:

```bash
python -m pip install -r requirements.txt
```

Then run the Streamlit app from the project root:

```bash
python -m streamlit run src/app/streamlit_app.py
```

After running the command, the app opens in the browser at:

```text
http://localhost:8501
```

---

## Requirements

The main Python packages used in this project are:

```text
pandas
numpy
scikit-learn
joblib
matplotlib
streamlit
```

---

## Project Structure

```text
predictive-maintenance-ai/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│       ├── preprocessor.joblib
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/
│   ├── final_baseline_random_forest.joblib
│   └── random_forest_baseline.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── reports/
│   ├── baseline_model_comparison.csv
│   ├── random_forest_feature_importance.csv
│   ├── random_forest_threshold_tuning_validation.csv
│   ├── random_forest_threshold_comparison.csv
│   ├── selected_baseline_model_info.json
│   │
│   └── figures/
│       ├── random_forest_confusion_matrix.png
│       ├── random_forest_feature_importance.png
│       ├── random_forest_precision_recall_curve.png
│       └── random_forest_roc_curve.png
│
├── src/
│   ├── app/
│   │   └── streamlit_app.py
│   │
│   ├── prediction/
│       ├── __init__.py
│       └── predict.py
│   
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Main Files

### `src/prediction/predict.py`

This file contains the reusable prediction pipeline. It loads the saved model and preprocessor, validates the input data, applies preprocessing, and returns the prediction results.

Main responsibilities:

```text
Load saved model
Load saved preprocessor
Validate required input columns
Ignore extra columns
Transform raw input data
Predict machine failure probability
Return prediction results
```

### `src/app/streamlit_app.py`

This file contains the Streamlit web app. It provides a user interface for manual prediction, batch CSV prediction, and model information.

Main responsibilities:

```text
Display app interface
Collect manual input values
Handle CSV upload
Show prediction results
Show model metrics
Show feature importance and evaluation figures
Allow prediction result download
```

---

## Example Use Case

A user can enter machine sensor values manually:

```text
Type: L
Air temperature [K]: 300
Process temperature [K]: 310
Rotational speed [rpm]: 1500
Torque [Nm]: 40
Tool wear [min]: 100
```

The app then returns a failure probability and classifies the machine as either:

```text
Normal
Failure risk
```

For batch prediction, the user can upload a CSV file containing multiple machines and download the prediction results as a new CSV file.

---

## Results

The final project shows that machine failure prediction can be performed using structured sensor data. The Random Forest model achieved strong ROC-AUC and useful recall for detecting failures.

The most important features were related to machine operation, especially rotational speed, torque, and tool wear. This makes sense because these features are directly connected to mechanical stress and machine usage.

---

## Limitations

This project is a machine learning portfolio project and not a production-ready industrial system.

Some limitations are:

```text
The dataset is synthetic
The failure class is highly imbalanced
The model was trained on one dataset only
Real industrial deployment would require more validation
The app does not connect to live machine sensors
```

---

## Future Improvements

Possible future improvements include:

```text
Deploying the Streamlit app online
Adding live sensor data input
Improving threshold selection based on business cost
Testing more advanced models
Adding model explainability with SHAP
Adding automated retraining
Adding unit tests for the prediction pipeline
Saving the preprocessor inside the models folder for cleaner artifact management
```

---

## Summary

This project demonstrates a complete machine learning workflow for predictive maintenance.

It includes:

```text
Data preprocessing
Model training
Model evaluation
Saved model artifacts
Reusable prediction pipeline
Interactive Streamlit app
Batch CSV prediction
GitHub-ready project structure
```

The final result is a practical machine learning application that can be used through a simple web interface instead of only through notebooks.