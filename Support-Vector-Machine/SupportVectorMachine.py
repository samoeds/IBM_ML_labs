import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests


'''Load Data From CSV File'''
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'

response = requests.get(path)

with open('cell_samples.csv', 'wb') as my_data:
    my_data.write(response.content)

# cell_df = pd.read_csv("cell_samples.csv", encoding='utf-8')
# print(cell_df.head())
#
# print("_______________________________")
#
# print(cell_df.shape)
#
# ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
# cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
# plt.show()
#
# print("_______________________________")
#
# '''Data pre-processing and selection'''
# print(cell_df.dtypes)

