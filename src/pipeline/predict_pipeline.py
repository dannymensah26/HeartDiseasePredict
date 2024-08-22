import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_data_frame(self):
        # Convert the data to a pandas DataFrame
        data = {
            'age': [self.age],
            'sex': [self.sex],
            'cp': [self.cp],
            'trestbps': [self.trestbps],
            'chol': [self.chol],
            'fbs': [self.fbs],
            'restecg': [self.restecg],
            'thalach': [self.thalach],
            'exang': [self.exang],
            'oldpeak': [self.oldpeak],
            'slope': [self.slope],
            'ca': [self.ca],
            'thal': [self.thal]
        }

        return pd.DataFrame(data)

class PredictPipeline:
    def __init__(self):
        # Load the model, preprocessors, etc.
        pass

    def predict(self, data):
        # Perform predictions using your trained model
        # For demonstration purposes, let's assume a binary classification model (0 or 1)
        # Replace this with your actual model inference code
        return [0]  # Example prediction (replace with actual prediction logic)












        
'''
class CustomData:
    def __init__(self,
                 age:float,
                 trestbps:float,
                 chol:float,
                 thalach:float,
                 oldpeak:float,
                 sex:str,
                 cp:str,
                 fbs:str,
                 restecg:str,
                 exang:str,
                 slope:str,
                 ca:str,
                 thal:str):
        
        self.age = age
        self.trestbps = trestbps
        self.chol = chol
        self.thalach = thalach
        self.oldpeak = oldpeak
        self.sex = sex
        self.cp = cp
        self.fbs = fbs
        self.restecg = restecg
        self.exang = exang
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'trestbps':[self.trestbps],
                'chol':[self.chol],
                'thalach':[self.thalach],
                'oldpeak':[self.oldpeak],
                'sex':[self.sex],
                'cp':[self.cp],
                'fbs':[self.fbs],
                'restecg':[self.restecg],
                'exang':[self.exang],
                'slope':[self.slope],
                'ca':[self.ca],
                'thal':[self.thal]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

'''