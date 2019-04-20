import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score



df = pd.read_csv('diabetes.csv')

print(df.columns)
df.head()

df.shape

print(df.groupby('Outcome').size())

import seaborn as sns
sns.countplot(df['Outcome'],label="Count")

df.info()

y = df.Outcome.values
x_data = df.drop(['Outcome'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
print(x.loc[1].values)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
		
    # record training set accuracy
    training_accuracy.append(knn.score(x_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(x_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
knn = KNeighborsClassifier(n_neighbors=9)
model = knn.fit(x_train,y_train)
joblib.dump(model, 'diabetes_knn_model.pkl', compress=True)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(x_test, y_test)))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(x_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(x_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(x_test, y_test)))

joblib.dump(logreg, 'diabetes_logreg_model.pkl', compress=True)




from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
dt=tree.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(x_test, y_test)))
joblib.dump(dt, 'diabetes_dt_model.pkl', compress=True)


from sklearn.svm import SVC
diabetes=SVC(kernel='linear')
svm=diabetes.fit(x_train,y_train)
predictions=diabetes.predict(x_test)
print(accuracy_score(y_test,predictions))
joblib.dump(svm, 'diabetes_svms_model.pkl', compress=True)



