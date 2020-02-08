from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_csv("C:\\Users\\pkdos\\Downloads\\apples_and_oranges.csv")
#print(data)
x=data[["Size","Weight"]]
y=data["Class"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
print(len(xtrain))

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(xtrain,ytrain)
y_pred=classifier.predict(xtest)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,ytest)
print ("Confusion Matrix : \n", cm)


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_pred,ytest))
