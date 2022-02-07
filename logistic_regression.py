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


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))