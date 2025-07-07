import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Paths to model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.sav')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.sav')

# Load model and scaler
diabetes_model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                           ['Diabetes Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    st.title('🩺 Diabetes Prediction using Machine Learning')

    # Input form layout
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # Example input tester
    with st.expander("🔍 Try Example Data"):
        example_choice = st.selectbox("Choose Sample Row:", ['None', 'Row 1', 'Row 2', 'Row 3', 'Row 4', 'Row 5'])
        example_data = {
            'Row 1': [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # diabetic
            'Row 2': [1, 85, 66, 29, 0, 26.6, 0.351, 31],   # not diabetic
            'Row 3': [8, 183, 64, 0, 0, 23.3, 0.672, 32],   # diabetic
            'Row 4': [1, 89, 66, 23, 94, 28.1, 0.167, 21],  # not diabetic
            'Row 5': [0, 137, 40, 35, 168, 43.1, 2.288, 33] # diabetic
        }
        if example_choice != 'None':
            values = list(map(str, example_data[example_choice]))
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age = values

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            # Collect and clean input
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            user_input = [x.strip() for x in user_input]

            if any(x == '' for x in user_input):
                st.warning("⚠️ Please fill in all the fields.")
            else:
                user_input = [float(x) for x in user_input]
                input_scaled = scaler.transform([user_input])
                diab_prediction = diabetes_model.predict(input_scaled)

                if diab_prediction[0] == 1:
                    diab_diagnosis = '✅ The person is diabetic'
                else:
                    diab_diagnosis = '✅ The person is not diabetic'

                st.success(diab_diagnosis)

        except ValueError:
            st.error("❌ Please enter valid numeric values in all fields.")
