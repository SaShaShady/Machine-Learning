import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
data=pd.read_csv("F:\\hdfc_stocks.csv",index_col=False)
print(data)
y=data["Price"]
x=data["Date"]
#plt.scatter(x,y)
#plt.show()
x_train=x.iloc[:1500]
x_test=x.iloc[1500:]

y_train=y.iloc[:1500]
y_test=y.iloc[1500:]

regs=linear_model.LinearRegression

#regs.fit(x_train,y_train)





#svm,decision trees
