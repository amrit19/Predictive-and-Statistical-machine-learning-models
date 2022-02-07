import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(scaled_features,eegdata['state'],
                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=3')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

    