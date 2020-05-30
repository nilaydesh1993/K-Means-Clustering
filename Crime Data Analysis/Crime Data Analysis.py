"""
Created on Mon Apr 20 14:36:23 2020
@author: DESHMUKH
KMEANS CLUSTERING 
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# ==============================================================================================
# Business Problem :- Perform Clustering for the crime data and identify the number of clusters.
# ==============================================================================================

crime = pd.read_csv("crime_data.csv")
crime = crime.rename({ 'Unnamed: 0' : 'city'},axis = 1)
crime.head()
crime.info()
crime.isnull().sum()
crime.shape
crime.columns

# Summary
crime.describe()

# Histogram
crime.hist()

# Scatter Plot
sns.pairplot(crime,diag_kind="kde")

# Standardization of Data (We can also use normalization)
from sklearn import preprocessing
crime_std = preprocessing.scale(crime.iloc[:,1:5])
crime_std = pd.DataFrame(crime_std,columns =['Murder', 'Assault', 'UrbanPop', 'Rape'] )

# Elbow curve 
sse = []
k_rng = range(2,15)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(crime_std)
    sse.append(km.inertia_)

# Scree plot or Elbow Curve
plt.plot(k_rng,sse,'o-',color = 'c');plt.ylabel('Sum of squared error');plt.xlabel('K')

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(crime_std)

# Getting the labels of clusters assigned to each row 
model.labels_ 

# Converting result into Dataframe or Series
cluster_labels = pd.DataFrame((model.labels_),columns = ['cluster'])

# Concating lable dataframe into original data frame
crime_final = pd.concat([cluster_labels,crime],axis=1)

# getting aggregate mean of each cluster
crime_final.iloc[:,1:].groupby(crime_final.cluster).mean()

# Creating a csv file 
#crime_final.to_csv("crime_final.csv",encoding="utf-8")

            #---------------------------------------------------------#






