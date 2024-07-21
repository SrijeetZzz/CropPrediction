import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action ='ignore')

data=pd.read_csv("/content/data (2).csv")
print(data)

sns.heatmap(data.isnull())

# summer crops


sc=data[(data['temperature']>30)&(data['humidity']>50)]['label'].unique()
print(sc)

# winter crop

wi=data[(data['temperature']<20)&(data['humidity']>30)]['label'].unique()
print(wi)

# monsoon crop


mo=data[(data['rainfall']>200)&(data['humidity']>50)]['label'].unique()
print(mo)

sns.pairplot(data,hue="label")

print(data)
x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,-1].values
print(y)

# KNN CLASSIFICATION

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=.3)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)

inp=[[40,10,20,10,80,10,100]]
yp=model.predict(inp)
print(yp)



from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


inp=[[40,10,20,10,80,10,100]]
yp=model.predict(inp)
print(yp)

# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)

inp=[[40,10,20,10,80,10,100]]
yp=model.predict(inp)
print(yp)