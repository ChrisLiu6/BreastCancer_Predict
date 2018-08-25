# Predict breast cancer based on UCI ML data base
# Download breast-cancer-wisconsin.data.txt

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

pd.set_option('display.max_columns', 20) # show hidden columns
df = pd.read_csv('breast-cancer-wisconsin.data.txt') # import data sheet

df.replace('?', int(-99999), inplace=True) # replace missing info as outlier
df['BareNuclei'] = df['BareNuclei'].astype(int) # fixed an bug where 6th column is string 
df.drop(['Sample_id'], 1, inplace=True) # drop ID column

x = np.array(df.drop(['class'],1)) 
y = np.array(df['class']) # class column as output

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier() # K nearst neighbors method
clf.fit(x_train, y_train) 
print('Accuracy:', clf.score(x_test, y_test)) # accuracy

number = int(input('How many patients: '))
data_list = []

# Putting patient info into a list
for i in range(number):
    case = input('Enter patient coefficients (space between 9 inputs): ')
    inputs = list(map(int,case.split(' '))) # takes a list then convert to array
    inputs = np.asarray(inputs)
    data_list.append(inputs)
    ##inputs = inputs.reshape(-1,1)

print('Prediction: ', clf.predict(data_list)) # displace list of predictions


