import streamlit as st
import pandas as pd
import pickle

# Load the trained model and feature names
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("feature_names.pkl", "rb") as feature_file:
    feature_names = pickle.load(feature_file)

# Streamlit UI
st.title("Vaccine Prediction App")
st.markdown("This app predicts the probability of taking the H1N1 vaccine based on user inputs.")

# User input fields
h1n1_worry = st.selectbox("Level of worry about H1N1 (0: Not worried, 1: A little worried, 2: Very worried)", [0, 1, 2])
h1n1_awareness = st.selectbox("Awareness about H1N1 (0: No, 1: Yes)", [0, 1])
antiviral_medication = st.selectbox("Taken antiviral medication? (0: No, 1: Yes)", [0, 1])
contact_avoidance = st.selectbox("Avoided contact? (0: No, 1: Yes)", [0, 1])
bought_face_mask = st.selectbox("Bought a face mask? (0: No, 1: Yes)", [0, 1])
wash_hands_frequently = st.selectbox("Wash hands frequently? (0: No, 1: Yes)", [0, 1])
is_health_worker = st.selectbox("Are you a health worker? (0: No, 1: Yes)", [0, 1])

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    'h1n1_worry': [h1n1_worry],
    'h1n1_awareness': [h1n1_awareness],
    'antiviral_medication': [antiviral_medication],
    'contact_avoidance': [contact_avoidance],
    'bought_face_mask': [bought_face_mask],
    'wash_hands_frequently': [wash_hands_frequently],
    'is_health_worker': [is_health_worker]
})

# Align input_data with training data columns
aligned_input_data = pd.DataFrame(0, index=[0], columns=feature_names)
for col in input_data.columns:
    aligned_input_data[col] = input_data[col]

# Predict button
if st.button("Predict"):
    # Predict vaccine uptake probability
    try:
        prediction = model.predict_proba(aligned_input_data)[0][1]  # Probability of vaccine uptake
        st.write(f"### Predicted Probability of Vaccine Uptake: {prediction * 100:.2f}%")
    except Exception as e:
        st.write(f"An error occurred during prediction: {e}")
