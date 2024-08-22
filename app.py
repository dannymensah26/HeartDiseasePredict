
'''
from flask import Flask, request, render_template, jsonify
#from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# Initialize Flask application
application = Flask(__name__)
app = application

# Configure logging
logging.basicConfig(level=logging.INFO)

# Route for the home page
@app.route('/')
def home_page():
    return render_template('home.html')

# Route for predicting data
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        return render_template('home.html')
    else:
        try:
            # Extract data from the form
            data = CustomData(
                age = int(request.form['age']),
                sex = request.form.get('sex'),
                cp = request.form.get('cp'),
                trestbps = int(request.form['trestbps']),
                chol = int(request.form['chol']),
                fbs = request.form.get('fbs'),
                restecg = int(request.form['restecg']),
                thalach = int(request.form['thalach']),
                exang = request.form.get('exang'),
                oldpeak = float(request.form['oldpeak']),
                slope = request.form.get('slope'),
                ca = int(request.form['ca']),
                thal = request.form.get('thal')

            )
               
            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Prediction DataFrame: {pred_df}")
             
            # Create a prediction pipeline and get results
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction results: {results}")

            return render_template('results.html', prediction=results)
                     
    
        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            return render_template('results.html', error=str(e))

# Main entry point
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)


'''


import streamlit as st
import pandas as pd
import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# Streamlit app title with subtitle
st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="wide")
st.title("Heart Disease Prediction App 💓")
st.subheader("An interactive app to predict the likelihood of heart disease based on various health metrics.")

# Sidebar for user input
st.sidebar.header("Input Features")

# Expander for advanced options
with st.sidebar.expander("Customize Your Input Features"):
    def user_input_features():
        # Organize inputs into columns for better layout
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=0, max_value=120, value=25)
            sex = st.selectbox('Sex', options=['male', 'female'])
            cp = st.selectbox('Chest Pain Type (cp)', options=['0', '1', '2', '3'])

        with col2:
            trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=200, value=120)
            chol = st.number_input('Serum Cholesterol (chol)', min_value=0, max_value=600, value=200)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=['0', '1'])

        with col3:
            restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', options=['0', '1', '2'])
            thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=150)
            exang = st.selectbox('Exercise Induced Angina (exang)', options=['0', '1'])

        oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', options=['0', '1', '2'])
        ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy (ca)', min_value=0, max_value=4, value=0)
        thal = st.selectbox('Thalassemia (thal)', options=['0', '1', '2', '3'])

        # Store inputs in a dictionary and return as a DataFrame
        data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        features = pd.DataFrame(data, index=[0])
        return features

# Get the user input
input_df = user_input_features()

# Display the user input
st.write("### Input Data", input_df)

# Prediction logic
if st.button("Predict"):
    try:
        # Convert input DataFrame to a format suitable for prediction
        custom_data = CustomData(
            age=int(input_df['age'][0]),
            sex=input_df['sex'][0],
            cp=input_df['cp'][0],
            trestbps=int(input_df['trestbps'][0]),
            chol=int(input_df['chol'][0]),
            fbs=input_df['fbs'][0],
            restecg=int(input_df['restecg'][0]),
            thalach=int(input_df['thalach'][0]),
            exang=input_df['exang'][0],
            oldpeak=float(input_df['oldpeak'][0]),
            slope=input_df['slope'][0],
            ca=int(input_df['ca'][0]),
            thal=input_df['thal'][0]
        )

        # Get DataFrame for prediction
        pred_df = custom_data.get_data_as_data_frame()
        logging.info(f"Prediction DataFrame: {pred_df}")

        # Run prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info(f"Prediction results: {results}")

        # Display the prediction result
        st.success(f"Prediction: {'Positive for Heart Disease' if results[0] else 'Negative for Heart Disease'}")

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        st.error(f"An error occurred: {e}")


# Footer or additional information with custom CSS
st.markdown("""
    ---
    <style>
    .css-1d391kg {background-color: #E5E5E5;}
    .css-1v3fvcr {padding: 2rem 2rem;}
    </style>
    **Note:** This prediction app is for educational purposes only and should not be used as a substitute for professional medical advice.
    """, unsafe_allow_html=True)




