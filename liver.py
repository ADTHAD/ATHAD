from sklearn.externals import joblib
import pandas as pd
import numpy as np
from flask import request

def prediction():
    if request.method == 'POST':
        model_logreg = joblib.load("models/Liver_prediction_model/logreg.pkl")

        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        Total_Bilirubin = float(request.form.get('Total_Bilirubin'))
        Direct_Bilirubin = float(request.form.get('Direct_Bilirubin'))
        Alkaline_Phosphotase = float(request.form.get('Alkaline_Phosphotase'))
        Alamine_Aminotransferase = float(request.form.get('Alamine_Aminotransferase'))
        Aspartate_Aminotransferase = float(request.form.get('Aspartate_Aminotransferase'))
        Total_Protiens = float(request.form.get('Total_Protiens'))

        Albumin = float(request.form.get('Albumin'))
        Albumin_and_Globulin_Ratio = float(request.form.get('Albumin_and_Globulin_Ratio'))
        target= 0
        
        data = [
            age,   
            sex,      
            Total_Bilirubin,      
            Direct_Bilirubin,      
            Alkaline_Phosphotase,      
            Alamine_Aminotransferase,   
            Aspartate_Aminotransferase,   
            Total_Protiens,  
            Albumin,      
            Albumin_and_Globulin_Ratio,
            target
        ]
        liver_df = pd.read_csv('models/Liver_prediction_model/indian_liver_patient.csv')
        gender = {'Male': 1,'Female': 2} 
        liver_df.Gender = [gender[item] for item in liver_df.Gender]
        liver_df.loc[582] = [i for i in data]
        X = liver_df.drop('Dataset', axis=1)
        X= (X - np.min(X)) / (np.max(X) - np.min(X)).values
        finX = X[['Total_Protiens','Albumin', 'Gender']]
        data_to_predict = finX.loc[582].tolist()
        print(data_to_predict)
        predicted_result = model_logreg.predict([data_to_predict])     

def predict_liver_stuff():
	if request.method == 'POST':
		try:
			return prediction()
		except:
			return 0
