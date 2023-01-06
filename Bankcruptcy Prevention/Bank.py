import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("bankruptcy-prevention.csv",delimiter=';')

df


# Encoding on the target variable 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:,-1])
df


Y=df.iloc[:,-1]
X=df.iloc[:,:-1]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

classifier=LogisticRegression()

classifier.fit(X_train.values,Y_train.values)


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train.values)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train.values)

print('Accuracy score of the training data : ', training_data_accuracy)


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test.values)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test.values)

print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (0.0,0.5,0.0,1.0,1.0,0.5)


# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('BANKCRUPT')
else:
  print('NON-BANKCRUPT')


import pickle

filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))

input_data = (0.0,0.5,0.0,1.0,1.0,0.5)


# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('BANKCRUPT')
else:
  print('NON-BANKCRUPT')