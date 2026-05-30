# predictive-maintenance-ai
## Streamlit Prediction App

This project includes a Streamlit web app that allows users to make machine failure predictions using the trained Random Forest baseline model.

The app takes raw machine sensor values as input, applies the saved preprocessing pipeline, and returns the predicted failure probability and predicted class.

### How the Prediction Pipeline Works

The prediction pipeline follows this workflow:

```text
Raw machine input
        ↓
Saved preprocessing pipeline
        ↓
Trained Random Forest model
        ↓
Failure probability and prediction label