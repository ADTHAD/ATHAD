from flask import request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

def prediction():
    model_svm = joblib.load("models/heart_disease_models/heart_svm_model.pkl")
    model_rfc = joblib.load("models/heart_disease_models/heart_rfc_model.pkl")

    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    chol = int(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = int(request.form.get('thalach'))

    exang = int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))

    slope = int(request.form.get('slope'))

    ca = int(request.form.get('ca'))

    thal = int(request.form.get('thal'))
    target= 0
    
    data = [
        age,   
        sex,      
        cp,      
        trestbps,      
        chol,      
        fbs,   
        restecg,   
        thalach,  
        exang,      
        oldpeak,
        slope,
        ca,
        thal,
        target
    ]
    df = pd.read_csv("models/heart_disease_models/heart.csv")
    lastindex=df.shape[0]+1

    #Insert the data to csv for preprocessing
    df.loc[lastindex] = [i for i in data]
    a=pd.get_dummies(df['cp'],prefix ="cp")
    b=pd.get_dummies(df['thal'],prefix ="thal")
    c=pd.get_dummies(df['slope'],prefix ="slope")
    frames=[df,a,b,c]
    
    df=pd.concat(frames,axis=1)
    df=df.drop(['cp','thal','slope'],axis=1)
    x_data =df.drop(['target'],axis=1)
    
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
    data_to_predict = x.loc[lastindex].tolist()
    #print(data_to_predict)
    predicted_result = model_svm.predict([data_to_predict])
    #print(predicted_result)
    if predicted_result[0]==0:
        return 'The person do not have heart disease'
    else:
        return 'The person has heart disease'
    
def predict_stuff():
    if request.method == 'POST':
        try:
            return prediction()
        except:
            return 0