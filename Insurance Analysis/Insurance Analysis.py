"""
Created on Wed Apr 22 20:40:34 2020
@author: DESHMUKH
KMEANS CLUSTERING
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# ==============================================================================================
# Business Problem :- Create clusters of persons falling in the same type in insurance dataset.
# ==============================================================================================

insurance = pd.read_csv("Insurance Dataset.csv")
insurance.head()
insurance.shape
insurance.isnull().sum()
insurance.columns = 'Pre_Paid', 'Age', 'Days_to_Renew', 'Claims_made', 'Income'
insurance.columns
insurance.info()

# Summary
insurance.describe()

# Boxplot
insurance.boxplot(notch='True',patch_artist=True,grid=False);plt.xticks(fontsize=6)

# Scatter Plot
sns.pairplot(insurance)

from sklearn.preprocessing import normalize
insurance_norm = insurance.copy()
insurance_norm.iloc[:,0:5] = normalize(insurance_norm.iloc[:,0:5]) 

# Elbow curve
sse = []
k_rng = range(2,15)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(insurance_norm)
    sse.append(km.inertia_)

# Scree plot or Elbow Curve
plt.plot(k_rng,sse,'d-',color = 'b');plt.ylabel('Sum of squared error');plt.xlabel('K')

# Selecting 6 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=6) 
model.fit(insurance_norm)

# Getting the labels of clusters assigned to each row 
model.labels_

# Converting numpy array into pandas Dataframe object 
md=pd.DataFrame((model.labels_),columns = ['cluster']) 

# Concating lable dataframe into original data frame
insurance_final = pd.concat([md,insurance],axis=1)
insurance_final.head()

# Getting aggregate mean of each cluster
insurance_final.iloc[:,1:].groupby(insurance_final.cluster).mean()

# Creating a csv file 
#insurance_final.to_csv("Insurance_final.csv",encoding="utf-8")

                    #-------------------------------------------------#




