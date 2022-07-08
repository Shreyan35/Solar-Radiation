# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:58:46 2021

@author: Debjani
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
df=pd.read_csv('SolarPrediction.csv')
print(df.shape)
print(df.describe())
print(df.info())
print(df.isna().sum())
sns.heatmap(df.corr(),annot=True)

c=df['Radiation']
c.hist()
plt.xlabel('Radiation')


c=df['Temperature'].head(350)
sns.lineplot(data=c)


c=df['Radiation'].head(350)
sns.lineplot(data=c)


c=df['Temperature']
c.hist(edgecolor='black')
plt.xlabel('Temperature')


c=df['Pressure']
c.hist(edgecolor='black')
plt.xlabel('Pressure')
plt.figure(figsize=(12,8))


c=df['Speed']
c.hist(edgecolor='black')
plt.xlabel('Speed')
plt.figure(figsize=(12,8))


c=df['WindDirection(Degrees)']
c.hist(edgecolor='black')
plt.xlabel('WindDirection(Degrees)')
plt.figure(figsize=(12,8))

df['TimeSunSet']=df['TimeSunSet'].apply(lambda x:x.split(":")[0]) # extracting the hour value from time
df['Sunset(hr)']=df['TimeSunSet']
df=df.drop('TimeSunSet',axis=1)


df['TimeSunRise']=df['TimeSunRise'].apply(lambda x:x.split(":")[0])
df['Sunrise(hr)']=df['TimeSunRise'].apply(lambda x:int(x))
df=df.drop('TimeSunRise',axis=1)

df['Sunset(hr)']=df['Sunset(hr)'].apply(lambda x:int(x))

#Extracting month from data
df['month']=df['Data'].apply(lambda x:x.split("/")[0])
df['month']=df['month'].apply(lambda x:int(x)) # converting the values into integer

#Extracting date from data
df['date']=df['Data'].apply(lambda x:x.split("/")[1])
df['date']=df['date'].apply(lambda x:int(x)) # converting the values into integer

#Time column is also related to radiation.It is the time corresponding to the radiation value.
#ie,what's the radiation at what time
df['Time(hr)']=df['Time'].apply(lambda x:x.split(":")[0])
df['Time(min)']=df['Time'].apply(lambda x:x.split(":")[1])
df['Time(sec)']=df['Time'].apply(lambda x:x.split(":")[2])


df['Time(hr)']=df['Time(hr)'].apply(lambda x:int(x))
df['Time(min)']=df['Time(min)'].apply(lambda x:int(x))
df['Time(sec)']=df['Time(sec)'].apply(lambda x:int(x))

df=df.drop(['Data','Time'],axis=1)

print(df.info())

plt.figure(figsize=(12,5))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")#annot is used for plotting the numerical value in the boxes

x=df.drop('Radiation',axis=1).values
y=df['Radiation'].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
x=np.array(x)
y=np.array(y)

#splitting the dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=8)




'''RANDOM FOREST'''
from sklearn.ensemble import RandomForestRegressor
rmc=RandomForestRegressor(n_estimators=30)
rmc.fit(x_train,y_train)
rmc_pred=rmc.predict(x_test)
#rmc.score(x_test, y_test)
#df_rmc= pd.DataFrame({'Real values':y_test,'Predicted values':rmc_pred})




#arr=np.array([[1475229326,48,30.46,60,177.39,3.62,18,6,7,29,23,55,26]])
#print("prediction using rmc",rmc.predict(arr))

pickle.dump(rmc,open('solar.pkl','wb'))


