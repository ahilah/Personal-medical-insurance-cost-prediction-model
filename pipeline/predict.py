import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# custom files
import columns

try:
    ds = pd.read_csv("D:/programming/information-technologies-of-smart-systems/calculation-and-graphic work/personal-medical-insurance-cost-prediction/data/new_data.csv")
    print('new data size', ds.shape)
except FileNotFoundError:
    print("Error: File not found!")
    exit()

# feature engineering
try:
    param_dict = pickle.load(open('D:/programming/information-technologies-of-smart-systems/calculation-and-graphic work/personal-medical-insurance-cost-prediction/src/param_dict.pickle', 'rb'))
except FileNotFoundError:
    print("Error: Parameter dictionary file not found!")
    exit()


# Categorical encoding
for column in columns.cat_columns[0:]:
    ds[column] = ds[column].map(param_dict['map_dicts'][column])


# Define target and features columns
X = ds[columns.X_columns]


# load the model and predict
try:
    model = pickle.load(open('D://programming//information-technologies-of-smart-systems//calculation-and-graphic work//personal-medical-insurance-cost-prediction//models//finalized_model.sav', 'rb'))
except FileNotFoundError:
    print("Error: Model file not found!")
    exit()

try:
    y_pred = model.predict(X)
    ds['charges_pred'] = model.predict(X)
    ds.to_csv('D:/programming/information-technologies-of-smart-systems/calculation-and-graphic work/personal-medical-insurance-cost-prediction/data/prediction_results.csv', index=False)
except Exception as e:
    print("Error occurred while predicting:", e)