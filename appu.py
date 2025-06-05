import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Title
st.title("🌍 Tanzania Climate Prediction")
st.markdown("""
This Web app predicts monthly temperature and rainfall in Tanzania using historical climate data (2000–2020)
""")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('tanzania_climate_data.csv')

@st.cache_resource
def load_model():
    return joblib.load('tanzania_climate_model.pkl')

df = load_data()
model = load_model()

# Sidebar input
with st.sidebar:
    st.header("Prediction Parameters")
    current_year = datetime.now().year
    year = st.number_input("Year", min_value=2000, max_value=current_year+10, value=current_year)
    
    month = st.selectbox(
        "Month",
        options=[
            (1, "January"), (2, "February"), (3, "March"), 
            (4, "April"), (5, "May"), (6, "June"),
            (7, "July"), (8, "August"), (9, "September"),
            (10, "October"), (11, "November"), (12, "December")
        ],
        format_func=lambda x: x[1],
        index=0
    )
    predict = st.button("Predict Climate")

# Main content
if predict:
    try:
        # One-hot encode the month
        month_num = month[0]
        month_columns = [f"Month_{i}" for i in range(1, 13)]
        month_data = {col: [1 if int(col.split("_")[1]) == month_num else 0] for col in month_columns}
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Year': [year],
            **month_data
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Display results
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Temperature", f"{prediction[0][0]:.1f} °C")
        with col2:
            st.metric("Predicted Rainfall", f"{prediction[0][1]:.1f} mm")

        # Historical context
        st.subheader("📈 Historical Context")
        historical = df[df['Month'] == month_num]
        avg_temp = historical['Average_Temperature_C'].mean()
        avg_rain = historical['Total_Rainfall_mm'].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Historical Avg Temperature", 
                      f"{avg_temp:.1f} °C", 
                      f"{prediction[0][0] - avg_temp:+.1f} °C vs average")
        with col2:
            st.metric("Historical Avg Rainfall", 
                      f"{avg_rain:.1f} mm", 
                      f"{prediction[0][1] - avg_rain:+.1f} mm vs average")

    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

# EDA Visualizations
st.subheader("🔍 Climate Trends Explorer")

tab1, tab2 = st.tabs(["Temperature Analysis", "Rainfall Analysis"])

with tab1:
    st.write("### Monthly Temperature Trends")
    yearly_avg = df.groupby('Year')['Average_Temperature_C'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='Year', y='Average_Temperature_C', data=yearly_avg, ax=ax)
    ax.set_title("Yearly Average Temperature")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x='Month', y='Average_Temperature_C', data=df, ax=ax)
    ax.set_title("Monthly Temperature Distribution")
    st.pyplot(fig)

with tab2:
    st.write("### Monthly Rainfall Trends")
    yearly_rain = df.groupby('Year')['Total_Rainfall_mm'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x='Year', y='Total_Rainfall_mm', data=yearly_rain, ax=ax)
    ax.set_title("Yearly Total Rainfall")
    ax.set_ylabel("Rainfall (mm)")
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
