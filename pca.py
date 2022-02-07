import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

eegdata=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/endsem evaluation seventh semester/pre_processed_data/eeg_data.csv")
#print(eegdata.info())

eegdataonly=pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/endsem evaluation seventh semester/pre_processed_data/eeg_dataonly.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(eegdata.drop('state',axis=1))
scaled_features = scaler.transform(eegdata.drop('state',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=eegdata.columns[:-1])
print(df_feat.head())

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scaled_features)
pca.fit(scaled_features)

x_pca = pca.transform(scaled_features)
print(scaled_features.shape)
print(x_pca.shape)


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=eegdata['state'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

pca.components_

df_comp = pd.DataFrame(pca.components_,columns=eegdata.drop('state',axis=1))

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


