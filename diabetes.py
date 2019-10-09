from flask import request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

def prediction():
	model_knn = joblib.load("models/diabetes_predictor_models/diabetes_knn_model.pkl")	
	Pregnancies = int(request.form.get('Pregnancies'))
	Glucose= int(request.form.get('Glucose'))
	BloodPressure= int(request.form.get('BloodPressure'))
	SkinThickness= int(request.form.get('SkinThickness'))
	Insulin= int(request.form.get('Insulin'))
	BMI= float(request.form.get('BMI'))
	DiabetesPedigreeFunction= float(request.form.get('DiabetesPedigreeFunction'))
	Age= int(request.form.get('Age'))

	
	Outcome= 0

	diabetes_data = [Pregnancies,
	Glucose,
	BloodPressure,
	SkinThickness,
	Insulin,
	BMI,
	DiabetesPedigreeFunction,
	Age,
	Outcome]
       
	df = pd.read_csv('models/diabetes_predictor_models/diabetes.csv')
	lastindex=df.shape[0]+1
	df.loc[lastindex] = [i for i in diabetes_data]

	y = df.Outcome.values
	x_data = df.drop(['Outcome'], axis = 1)
	x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
	##print(x.loc[1].values)

	data_to_predict = x.loc[lastindex].tolist()
	#print(data_to_predict)
	predicted_result = model_knn.predict([data_to_predict])
	#print(predicted_result)
	if predicted_result[0]==0:
		result='The person does not have diabetes'
	else:
		result = 'The person have diabetes'

	test_accuracy=model_knn.score([data_to_predict],predicted_result)
	
	return [result,test_accuracy]

def predict_diabetes_stuff():
	if request.method == 'POST':
    		try:
    			return prediction()
    		except:
    			return 0

		  
