import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

eegdata=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/Eighth Semester/Midsem Evaluation/1.csv")
#print(eegdata.info())

#eegdataonly=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/endsem evaluation seventh semester/pre_processed_data/eeg_dataonly.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(eegdata.drop('state',axis=1))
scaled_features = scaler.transform(eegdata.drop('state',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=eegdata.columns[:-1])
print(df_feat.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, np.ravel(eegdata['state']), test_size=0.30, random_state=100)

print(X_train)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
