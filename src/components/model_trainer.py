
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
from src.utils import print_evaluated_results
from src.utils import model_metrics



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

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}')


             # Hyperparameter tuning on Random Forest Classifier
            # Initialize Random Forest Classifier
            rf = RandomForestClassifier()

             # Creating the hyperparameter grid on RF
            param_grid_rf = {'criterion': ['gini', 'entropy', 'log_loss'],
                    'n_estimators': [10, 50, 100, 200, 300],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]}

            # Instantiate RandomizedSearchCV object
            rscv = RandomizedSearchCV(rf, param_grid_rf, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

            # Fit the model
            rscv.fit(X_train, y_train)

            # Print the tuned parameters and score
            print(f'Best Random Forest parameters: {rscv.best_params_}')
            print(f'Best Random Forest Score: {rscv.best_score_}') 
            print('\n====================================================================================\n')

            # Retrieve the best model
            best_rf = rscv.best_estimator_

            logging.info('Hyperparameter tuning complete for Random Forest')


             # Hyperparameter tuning on Decision Tree Classifier
            logging.info('Hyperparameter tuning started for Decision Tree')

            # Initialize Decision Tree Classifier
            dt = DecisionTreeClassifier()

             # Creating the hyperparameter grid for DT
            param_grid_dt= {'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_leaf_nodes': [None, 10, 20, 30, 40, 50]}

            # Instantiate GridSearchCV object
            grid = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)

            # Print the tuned parameters and score
            print(f'Best Decision Tree Parameters : {grid.best_params_}')
            print(f'Best Decision Tree Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_dt = grid.best_estimator_

            logging.info('Hyperparameter tuning complete for Decision Tree')

          
             # Hyperparameter tuning on Support Vector Classifier

            logging.info('Hyperparameter tuning started for Support Vector Classifier')
            # Initialize Support Vector Classifier
            svc = SVC()

            param_grid_sv = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100, 1000], 
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4],
                    'coef0': [0.0, 0.1, 0.5, 1.0]}

            # Instantiate GridSearchCV object
            grid = GridSearchCV(svc, param_grid_sv, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)

            # Print the tuned parameters and score
            print(f'Best Support Vector Classifier Parameters : {grid.best_params_}')
            print(f'Best Support Vector Classifier Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_svc = grid.best_estimator_

            logging.info('Hyperparameter tuning complete for Support Vector Classifier')
             

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model pickle file saved')


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model pickle file saved')

            
            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            

            precision, recall, f1, report = model_metrics(y_test, predicted)

            logging.info(f'Precision : {precision}')
            logging.info(f'Recall : {recall}')
            logging.info(f'F1 Score : {f1}')
            logging.info(f'Classification Report : {report}')
            logging.info(f'Accuracy Score : {accuracy_score}')
            logging.info('Final Model Training Completed')

            return accuracy, precision, recall, f1, report,accuracy_score
           
        except Exception as e:
            raise CustomException(e,sys)


      






                 
           
