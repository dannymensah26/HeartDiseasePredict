## HEART DISEASE PREDICTION DEPLOYMENT ON STREAMLIT

### Overview

A simple web application which uses Machine Learning algorithm to predict the heart condition of a person by providing some inputs about the person health like age, gender, blood pressure, cholesterol level etc built using streamlit and deployed on Streamlit.

### Motivation

As being a Data and ML enthusiast I have tried many different projects related to the subject but what I have realised is that Deploying your machine learning model is a key aspect of every ML and Data science project. Everything thing I had studied or been taught so far in my Data science and ML journey had mostly focused on defining problem statement followed by Data collection and preparation, model building and evaluation process which is of course important for every ML/DS project but what if I want different people to interact with my models, how can I make my model available for end-users? I can't send them jupyter notebooks right!. That's why I wanted to try my hands on complete end-to-end machine learning project.

    To View the Deployed Application, click on the link given below : Heart Disease Prediction App - https://heartdiseasepredict-nz8hrvjjkwcswccqmtm4yj.streamlit.app/

    To get the Code for Exploratory data analysis/visualisations, different algorithms used and the model evaluation, click on the link mentioned below : Link of jupyter notebook - 
EDA:
https://github.com/dannymensah26/HeartDiseasePredict/blob/main/notebook/EDA.ipynb

Model Building:
https://github.com/dannymensah26/HeartDiseasePredict/blob/main/notebook/ModelBuilding.ipynb

https://github.com/user-attachments/assets/67a9b933-1345-4439-a0d2-c9507c3957f2




### A Demo of the Web App :

Heart_disease


#### Technical Aspect

This Project is mainly divided into two parts:

    Exploring the dataset and traning the model using Sklearn.
    Building and hosting a streamlit App on Streamlit

### About the repository Structure :

    Project consist app.py script which is used to run the application and is engine of this app. contians API that gets input from the user and computes a predicted value based on the model.
    prediction.py contains code to build and train a Machine learning model.
    templates folder contains two files main.html and result.html which describe the structure of the app and the way this web application behaves. These files are connected with Python via Flask framework.
    static folder contains file style.css which adds some styling and enhance the look of the application.

### Installation

The Code is written in Python 3.9. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

pip install -r requirements.txt 

To clone the repository

git clone https://github.com/dannymensah26/HeartDiseasePredict.git

### Run

To Run the Application

stream run app.py

### Deployement on Streamlit

Install Streamlit as this makes it easy to create and manage your Streamlit apps directly from the terminal. You can download it from here.

next step would be to follow the instruction given on Heroku Documentation to deploy a web app.


### Future work

    Improve model performance using Adaboost or XGBoost Gradient Algorithm
    Add more better styling to the user interface.

Some Useful Resources

    Streamlit Quickstart Documentation : https://docs.streamlit.io/
