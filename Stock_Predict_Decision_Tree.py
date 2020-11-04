from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

#Load the data
from google.colab import files
files.upload()

#Store the data into a variable
df = pd.read_csv('')

df = df.set_index(pd.DateTimeIndex(df['Date'].values))
#Give the index a name
df.index.name = 'Date'

#Manipulate the data
#Create target column
df['Price_Up'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df.drop(columns=['Date'])

#Split the dataset into a feature and target dataset
X = df.iloc[:, 0:df.shape[1] - 1].values
Y = df.iloc[:, df.shape[1]-1].values

#Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Create and train the model (DecisionTreeClassifier)
tree = DecisionTreeClassifier().fit(X_train, Y_train)

#Show the model's performance on test dataset
print(tree.score(X_test, Y_test))

#Show the model's prediction
tree_predict = tree.predict(X_test)
print(tree_predict)

#Show the actual values in test dataset
Y_test
