import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os, base64

st.set_page_config(page_title="Insurance cost Predictor", layout='wide')


# --- BACKGROUND IMAGE ---
import base64

with open("assets/backImg.png", "rb") as img:
    image = base64.b64encode(img.read()).decode()

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}


</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
  return joblib.load('models/insurance_model.pkl')

try:
  model_data = load_model()
  model = model_data['model']
  encoders = model_data['encoders']
except FileNotFoundError:
  st.error("Model not found. Please run `python train_model.py` first.")
  st.stop()

# Graphs
@st.cache_data
def load_data():
  return pd.read_csv('data/insurance.csv')

df = load_data()

st.title("Health Insurance Cost Predictor")
st.markdown("Predict your insurance charges based on personal attributes")

# Tabs for Analysis/Form
tab1, tab2 = st.tabs(["Data Analysis", "Prediction Form"])

# Tab 1: Data Analysis
with tab1:
  st.header("Exploratory Data Analysis")

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax1, color='teal')
    st.pyplot(fig1)

    st.subheader("BMI Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['bmi'], kde=True, ax=ax2, color='orange')
    st.pyplot(fig2)

  with col2:
    st.subheader("Smoker vs Non-Smoker")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='smoker', hue='smoker', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Region Count")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=df, x='region', palette='viridis', hue='region', ax=ax4)
    st.pyplot(fig4)

    st.subheader("Charges Distribution")
    fig5, ax5 = plt.subplots()
    sns.histplot(df['charges'], kde=True, ax=ax5, color='purple')
    st.pyplot(fig5)

with tab2:
  st.header("Enter Details")

  with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
      age = st.slider("Age", min_value=18, max_value=64, value=30)
      bmi = st.slider("BMI", min_value=16.0, max_value=53.2, value=30.0, step=0.1)

    with col2:
      sex = st.selectbox("Sex", options = ['male', 'female'])
      children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, step=1)

    with col3:
      smoker = st.selectbox("Smoker", options=['yes', 'no'])
      region = st.selectbox("region", options=['southeast', 'southwest', 'northeast', 'northwest'])

    submitted = st.form_submit_button("Calculate Insurance Cost")

  if submitted:
    input_dict = {
      'age' : age,
      'sex' : sex,
      'bmi' : bmi,
      'children' : children,
      'smoker' : smoker,
      'region' : region
    }

    # Encode categorical Input
    input_df = pd.DataFrame([input_dict])

    input_df['sex'] = input_df['sex'].map(encoders['sex'])
    input_df['smoker'] = input_df['smoker'].map(encoders['smoker'])
    input_df['region'] = input_df['region'].map(encoders['region'])

    # Ensure column order matches training
    input_df = input_df[model_data['features_name']]

    # Predition
    predict = model.predict(input_df)

    st.success(f"The estimate insurance cost is: ${predict[0]:,.2f}")

    if predict[0] > df['charges'].mean():
      st.warning("This estimate is above average.")
    else:
      st.info("This estimate is below average.")