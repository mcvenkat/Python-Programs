from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())  # To check if our data is loaded correctly

#Converting the data
le = preprocessing.LabelEncoder()

#fit.transform to take a list and return an array
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"  #optional
#Recombine the data
X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

#Split the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

Training a KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

#Testing the model
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

#Looking at neighbors
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # Now we will we see the neighbors of each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
