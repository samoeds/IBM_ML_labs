'''Customer Segmentation with K-Means'''
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

'''load data'''
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'

# response = requests.get(path)
#
# with open('Cust_Segmentation', 'wb') as my_data:
#     my_data.write(response.content)


cust_df = pd.read_csv('Cust_Segmentation', encoding='utf-8')
print(cust_df.head(), cust_df.shape)
print("_______________________________")

'''Preprocessing'''

df = cust_df.drop('Address', axis=1)
print(df.head())
print("_______________________________")

'''Normalizing over the standard deviation'''

from sklearn.preprocessing import StandardScaler
X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)
print("_______________________________")

'''Modeling'''
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
print("_______________________________")
labels = k_means.labels_
print(labels)
print("_______________________________")

'''Assign the labels to each row in the dataframe.'''
df["Clus_km"] = labels
df.head(5)

'''check the centroid values by averaging the features in each cluster'''
df.groupby('Clus_km').mean()

'''distribution of customers based on their age and income'''
area = np.pi * (X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float32), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float32))