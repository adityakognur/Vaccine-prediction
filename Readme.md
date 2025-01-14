# Vaccine Prediction App

This is a Streamlit-based web application that predicts the probability of taking the H1N1 vaccine based on user input. The model is built using machine learning techniques and trained on user behavior data related to H1N1 concerns.

## Project Overview

The Vaccine Prediction App allows users to provide inputs regarding their concerns and behaviors related to the H1N1 virus. Based on the inputs, the app predicts the likelihood of the user taking the H1N1 vaccine. The application utilizes a trained machine learning model, which has been saved as a `pickle` file, for making predictions.

## Features

- **User Input Fields**: Users can input various behaviors and concerns related to H1N1:
  - Level of worry about H1N1
  - Awareness about H1N1
  - Whether antiviral medication was taken
  - Whether contact avoidance and mask usage were adopted
  - Frequency of handwashing
  - Whether the user is a health worker
- **Prediction Output**: Once the inputs are submitted, the app predicts the probability of vaccine uptake and displays the result to the user.

## How to Run the Application

1. **Install Dependencies**:
    Make sure you have Python installed. Then, install the required dependencies:
    ```bash
    pip install streamlit pandas scikit-learn
    ```

2. **Place Model Files**:
    - `model.pkl`: A pickled machine learning model trained for vaccine prediction.
    - `feature_names.pkl`: A pickled file containing the feature names used in the model.

3. **Run the Application**:
    Navigate to the project folder and run the following command to start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    This will launch the app in your default web browser.

## Files in the Project

- `app.py`: The main Python script that runs the Streamlit web application.
- `model.pkl`: The trained machine learning model used to predict vaccine uptake probability.
- `feature_names.pkl`: A file containing the list of features used in the trained model.

## How the Model Works

The model is trained on a dataset with information about user behavior related to the H1N1 virus. It takes into account features such as:
- Worry about H1N1
- Awareness of H1N1
- Whether antiviral medications were taken
- Whether the individual avoided contact or wore a face mask
- Frequency of handwashing
- Health worker status

The model outputs the probability of the user accepting the H1N1 vaccine, which is displayed on the app after the user provides their inputs.
