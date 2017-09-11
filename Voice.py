
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

vc=pd.read_csv('voice.csv')
vc.head()
vc.columns
#sns.heatmap(vc.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.pairplot(vc)

X= vc[['meanfreq', 'skew', 'kurt','IQR','Q25', 'Q75',
       'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
       'meandom', 'mindom', 'maxdom','dfrange' ]]
#, 'sd', 'median','Q25', 'Q75', 'modindx','dfrange'
y=vc['label1']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print('logistic regression report')
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# conducting decision tress
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)
print('decision tree report')
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# conducting random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print('Random forest report')
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))


