import sklearn
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


"""df=pd.read_csv("diabetes.csv")

y = df.Outcome.values
x_data = df.drop(['Outcome'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
"""
loaded_model=joblib.load('diabetes_knn_model.pkl')
test=[0.352941,0.743719,0.590164,0.353535,0.000000,0.500745,0.234415,0.483333]
test1=[0.35294117647058826, 0.7437185929648241, 0.5901639344262295, 0.23333333333333334, 0.0, 0.5007451564828614, 0.23441502988898377, 0.6125]




print(loaded_model.predict([test]))
print(loaded_model.predict([test1]))
#print(y_test)
