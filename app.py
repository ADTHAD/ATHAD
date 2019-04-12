from flask import Flask, render_template, jsonify, redirect, request
import pandas as pd
from sklearn.externals import joblib

import numpy as np

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html") 

@app.route("/scrape", methods=['GET']) 
def scrape():
    listings = mongo.db.listings
    listings_data = scrape_craigslist.scrape()
    listings.update(
        {},
        listings_data,
        upsert=True
    )
    return redirect("http://localhost:5500/", code=302)

@app.route('/get-user-data', methods=['POST'])
def predict_stuff():
    if request.method == 'POST':
        model_svm = joblib.load('heart_svm_model.pkl')
        model_rfc = joblib.load('heart_rfc_model.pkl')
        print('-----line 27--------')
        

        age = int(request.form.get('age'))

        print('line 31')

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
        df = pd.read_csv("heart.csv")
        #Insert the data to csv for preprocessing
        df.loc[303] = [i for i in data]
        a=pd.get_dummies(df['cp'],prefix ="cp")
        b=pd.get_dummies(df['thal'],prefix ="thal")
        c=pd.get_dummies(df['slope'],prefix ="slope")
        frames=[df,a,b,c]
        
        df=pd.concat(frames,axis=1)
        df=df.drop(['cp','thal','slope'],axis=1)
        x_data =df.drop(['target'],axis=1)
        
        x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
        data_to_predict = x.loc[303].tolist()
        
        predicted_result = model_svm.predict(data_to_predict)
        print(predicted_result)
        if predicted_result[0]==0:
        	result='The person do not have heart disease'
        else:
        	result = 'The person has heart disease'
        

        return render_template("index.html", pred=result) 


if __name__ == "__main__":
    app.run()
