import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\pkdos\\Downloads\\User_data.csv",index_col=False)
#print(data)
x=data[["Age","EstimatedSalary"]]
y=data["Purchased"]
#print(x)
#print(y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)

regr=LogisticRegression()
regr.fit(x,y)

ytest=regr.predict(xtest)
print(ytest)
print(xtest)
#plt.scatter(xtest,ytest)
#plt.show()
print(regr.score(xtest,ytest))

