import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🌦️ Weather Prediction App")
st.write("Enter the weather features below:")

precip = st.number_input("Precipitation", value=0.25)
temp_max = st.number_input("Max Temperature", value=17.0)
temp_min = st.number_input("Min Temperature", value=11.0)
wind = st.number_input("Wind", value=2.5)

if st.button("Predict Weather"):
    input_df = pd.DataFrame([{
        "precipitation": precip,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "wind": wind
    }])
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    predicted_label = le.inverse_transform([prediction])[0]
    probs = model.predict_proba(input_scaled)[0]

    st.success(f"🌤️ Predicted Weather: {predicted_label}")
    st.subheader("📊 Probabilities:")
    for i, prob in enumerate(probs):
        label = le.inverse_transform([i])[0]
        st.write(f"{label}: {prob * 100:.2f}%")
