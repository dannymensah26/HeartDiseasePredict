
import os
import sys

import numpy as np 
import pandas as pd
import dill

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import logging

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            #para=param[list(models.keys())[i]]

            #gs = GridSearchCV(model,para,cv=3)
            #gs.fit(X_train, y_train)
            #model.set_params(**gs.best_params_)

            # Train model
            model.fit(X_train,y_train)

            # Predict Training data
            y_train_pred = model.predict(X_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get accuracy scores for train and test data
            train_model_score = accuracy_score(y_train,y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
'''
def model_metrics(true, predicted):
    try :
        precision = precision_score(true, predicted, average='weighted')
        recall = recall_score(true, predicted, average='weighted')
        f1 = f1_score(true, predicted, average='weighted')
        report = classification_report(true, predicted, average='weighted' )
        return precision, recall, f1, report

    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)
    

def print_evaluated_results(X_train,y_train,X_test,y_test,model):
    try:
        ytrain_pred = model.predict(X_train)
        ytest_pred = model.predict(X_test)

    
        # Evaluate Train and Test dataset
        model_train_precision, model_train_recall, model_train_f1,model_train_report = model_metrics(y_train, ytrain_pred)
        model_test_precision, model_test_recall, model_test_f1,model_test_report = model_metrics(y_test, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Precision: {:.4f}".format(model_train_precision))
        print("- Recall: {:.4f}".format(model_train_recall))
        print("- F1 Score: {:.4f}".format(model_train_f1))
        print("- Classification Report: {:.4f}".format(model_train_report))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Precision: {:.4f}".format(model_test_precision))
        print("- Recall: {:.4f}".format(model_test_recall))
        print("- F1 Score: {:.4f}".format(model_test_f1))
        print("- Classification Report: {:.4f}".format(model_test_report))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)

'''
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)


