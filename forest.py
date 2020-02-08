from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv("C:\\Users\\pkdos\\Desktop\\petrol_consumption.csv")
print(data.head())
x=data.iloc[:,0:4].values
y=data.iloc[:,4].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
regr=RandomForestRegressor(n_estimators=20,random_state=1)
regr.fit(x_train,y_train)
y_pred=regr.predict(x_test)

print(pd.DataFrame({'actual':y_test,'predicted':y_pred}))
plt.scatter(y_test,y_pred)
plt.show()