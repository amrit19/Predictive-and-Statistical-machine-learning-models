import pandas as pd
import plotly.express as px
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import pi
df = pd.read_csv('D:/NIT MY STUDY/FINAL YEAR PROJECT/endsem evaluation seventh semester/1_11.csv')
print(df.head())
print(df.plot())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_features = scaler.transform(df)
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())