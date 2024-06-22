from flask import Flask, request, Response, json
from flask import jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

#load data
#dataLoc ="./Sample_Data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
dataLoc ="./churnData/processed_dataset_subset.csv"
df4 = pd.read_csv(dataLoc,sep = ',')

# Split the data into features and target variable
X = df4.drop('Churn', axis=1)
Y = df4['Churn']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

# Standardize the features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Train a logistic regression model
#model = LogisticRegression()
#model.fit(X_train, Y_train)


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, Y_train)

# Evaluate the model
#Y_pred = model.predict(X_test)
Y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

#print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}')
#print(classification_report(Y_test, Y_pred))

# Save the model and scaler to disk
joblib.dump(rf_classifier, 'churn_model.pkl')

#create flask instance
from flask import Flask
app = Flask(__name__)


# Load the trained model and scaler
model = joblib.load('churn_model.pkl')


#==========================
#create api

@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)

    data_Senior = np.array([data["SeniorCitizen"]])
    data_Senior = np.reshape(data_Senior, (1, -1))

    data_tenure = np.array([data["tenure"]])
    data_tenure = np.reshape(data_tenure, (1, -1))

    data_gender = np.array([data["gender_Male"]])
    data_gender = np.reshape(data_gender, (1, -1))

    data_dependents = np.array([data["Dependents_Yes"]])
    data_dependents = np.reshape(data_dependents, (1, -1))

    data_final = np.column_stack((data_Senior, data_tenure, data_gender,data_dependents))
    #make predicon using model
    prediction = model.predict(data_final)
    return Response(json.dumps(prediction[0],default=str))