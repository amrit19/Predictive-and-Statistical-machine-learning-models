import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

eegdata=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/Eighth Semester/Midsem Evaluation/1.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(eegdata.drop('state',axis=1))
scaled_features = scaler.transform(eegdata.drop('state',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=eegdata.columns[:-1])
print(df_feat.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, np.ravel(eegdata['state']), test_size=0.30, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(eegdata.columns[1:])
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 

#comparing with random forests

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train) 

rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


