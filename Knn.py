from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
data=pd.read_csv("C:\\Users\\pkdos\\Downloads\\salary.csv")
x=data["Experience"]
y=data["Actual_Salary"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
import numpy as np

from sklearn.model_selection import train_test_split

'''array = data.values
X = array[:,:2]
y = array[:,2]
data.shape
print(X)'''
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 10)
knnr.fit(xtrain, ytrain)
