import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import seaborn as sns
import math

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
mpl.style.use('seaborn')

eegdata=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/Eighth Semester/Midsem Evaluation/1.csv")
#print(eegdata.info())

eegdataonly=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/Eighth Semester/Midsem Evaluation/1.csv")

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
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

df_cm = pd.DataFrame(confusion_matrix(y_test,pred), 
  index = ['EOS', 'IOS', 'MOS','ROS','SROS','TOS'],
  columns = ['EOS', 'IOS', 'MOS','ROS','SROS','TOS'])

fig = plt.figure()

plt.clf()

ax = fig.add_subplot(111)
ax.set_aspect(1)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, fmt='.2f', cmap=cmap)

res.invert_yaxis()

plt.yticks([0.5,1.5,2.5,3.5,4.5,5.5],[ 'EOS', 'IOS', 'MOS','ROS','SROS','TOS'],va='center')

plt.title('Confusion Matrix')

plt.savefig('D:/Python/confusion_matrix_KNN_EEG.png', dpi=100, bbox_inches='tight' )

plt.close()


