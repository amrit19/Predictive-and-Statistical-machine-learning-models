import pandas as pd
df = pd.read_csv("D:/NIT MY STUDY/FINAL YEAR PROJECT/Eighth Semester/Midsem Evaluation/1.csv")
df.head()
import pandas as pd
import plotly.express as px

df = pd.read_csv('D:/NIT MY STUDY/FINAL YEAR PROJECT/Eighth Semester/Midsem Evaluation/1.csv')

fig = px.line(df, x = 'g1', y = 'f1', title='Apple Share Prices over time (2014)')
fig.show()