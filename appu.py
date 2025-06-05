import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model
model = joblib.load('tanzania_climate_model.pkl')

# Title
st.title("🌍 Tanzania Climate Prediction & Trends")
st.markdown("""
This Streamlit app allows you to:
- Explore Tanzania climate data trends 📈
- Predict future temperature & rainfall 🌧️🌡️
""")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
    month = st.selectbox("Month", range(1, 13), index=0)
    prev_temp = st.number_input("Previous Month Avg Temperature (°C)", min_value=10.0, max_value=40.0, value=25.0)
    prev_rain = st.number_input("Previous Month Total Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    predict = st.button("Predict Climate")

# Make prediction
if predict:
    input_df = pd.DataFrame({
        'Year': [year],
        'Month': [month],
        'Previous_Temperature': [prev_temp],
        'Previous_Rainfall': [prev_rain]
    })

    prediction = model.predict(input_df)

    st.subheader("📊 Prediction Result")
    st.write(f"Predicted Avg Temperature: **{prediction[0][0]:.2f} °C**")
    st.write(f"Predicted Total Rainfall: **{prediction[0][1]:.2f} mm**")

    st.line_chart(pd.DataFrame({
        'Month': ['Previous', 'Predicted'],
        'Temperature': [prev_temp, prediction[0][0]],
        'Rainfall': [prev_rain, prediction[0][1]]
    }).set_index('Month'))

# EDA visualizations from notebook
st.subheader("🔍 Climate Trends in Tanzania (2000–2020)")

df = pd.read_csv('tanzania_climate_data.csv')
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))

tab1, tab2, tab3 = st.tabs(["Temperature Trend", "Rainfall Trend", "Monthly Patterns"])

with tab1:
    st.write("### Average Temperature Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='Date', y='Average_Temperature_C', data=df, ax=ax)
    ax.set_title("Average Temperature in Tanzania")
    st.pyplot(fig)

with tab2:
    st.write("### Total Rainfall Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='Date', y='Total_Rainfall_mm', data=df, ax=ax)
    ax.set_title("Total Rainfall in Tanzania")
    st.pyplot(fig)

with tab3:
    st.write("### Temperature & Rainfall by Month")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x='Month', y='Average_Temperature_C', data=df, ax=ax)
    ax.set_title("Monthly Temperature Distribution")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x='Month', y='Total_Rainfall_mm', data=df, ax=ax)
    ax.set_title("Monthly Rainfall Distribution")
    st.pyplot(fig)

# Footer
st.markdown("""
---
🧠 *Model based on historical data from 2000 to 2020. For educational and exploratory purposes.
     For more accurate predictions, more sophisticated models and additional climate factors would be needed
    Thank you for understanding*
""")
