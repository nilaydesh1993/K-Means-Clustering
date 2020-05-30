"""
Created on Sun Apr 19 19:41:27 2020
@author: DESHMUKH
KMEANS CLUSTERING
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
pd.set_option('display.max_column',None)

# ===================================================================================
# Business Problem :- Obtain optimum number of clusters for the airlines data. 
# ===================================================================================

airline1 = pd.read_excel("EastWestAirlines.xlsx",sheet_name = 'data')
airline = airline1
airline = airline.drop('ID#',axis = 1)
airline.head()
airline.columns
airline.isnull().sum()
airline.info()
airline.shape

# Summary 
airline.describe()

# Histogram
airline.hist()

# Scatter Plot
sns.pairplot(airline)

# Normalization of Data (beacuse it contain Binary value)
from sklearn.preprocessing import normalize
airline.iloc[:,0:10] = normalize(airline.iloc[:,0:10])
airline.head()

# Elbow curve 
sse = []
k_rng = range(2,15)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(airline)
    sse.append(km.inertia_)

# Scree plot or Elbow Curve
plt.plot(k_rng,sse,'H--',color = 'G');plt.ylabel('Sum of squared error');plt.xlabel('K')

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(airline)

# Getting the labels of clusters assigned to each row 
model.labels_

# Converting numpy array into pandas Dataframe object 
md=pd.DataFrame((model.labels_),columns = ['cluster']) 

# Concating lable dataframe into original data frame
airlinefinal = pd.concat([md,airline1],axis=1)
airlinefinal.head()

# Getting aggregate mean of each cluster
airlinefinal.iloc[:,2:].groupby(airlinefinal.cluster).mean()

# Creating a csv file 
#airlinefinal.to_csv("Airlinefinal.csv",encoding="utf-8")

                    #-------------------------------------------------#
