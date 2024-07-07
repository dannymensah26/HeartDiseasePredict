
import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models
#from src.utils import print_evaluated_results
#from src.utils import model_metrics



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector": SVC()

            }
      

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
             
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model pickle file saved')

            
            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy

            #accuracy_score, precision, recall, f1, report = model_metrics(y_test, predicted)

            #logging.info(f'Precision : {precision}')
            #logging.info(f'Recall : {recall}')
            #logging.info(f'F1 Score : {f1}')
            #logging.info(f'Classification Report : {report}')
            logging.info(f'Accuracy Score : {accuracy_score}')
            logging.info('Final Model Training Completed')

            #return precision, recall, f1, report,accuracy_score
           
        except Exception as e:
            raise CustomException(e,sys)


      






                 
           
