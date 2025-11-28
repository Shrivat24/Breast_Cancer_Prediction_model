import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib 


print(os.getcwd())  # shows your working folder

# use forward slashes (or a raw string) to avoid accidental escape sequences like '\r'
data=pd.read_csv(r'C:\Users\ADMIN\Desktop\MY PROJECT\Deployed projects\Breast_Cancer_Prediction_Model\data\raw\Breast_cancer_data.csv')

x=data.drop('diagnosis',axis=1)
y=data['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Random forest
rf=RandomForestClassifier()
rf=rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

os.makedirs("models", exist_ok=True)
model_path = "models/breast_cancer_rf_model.pkl"
joblib.dump(rf, model_path)
print(f"Model saved successfully at: {model_path}")