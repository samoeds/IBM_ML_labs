import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests


'''Load Data From CSV File'''
# path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
#
# response = requests.get(path)
#
# with open('cell_samples.csv', 'wb') as my_data:
#     my_data.write(response.content)

cell_df = pd.read_csv("cell_samples.csv", encoding='utf-8')
print(cell_df.head())

print("_______________________________")

print(cell_df.shape)

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',
                                               label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow',
                                          label='benign', ax=ax);
# plt.show()

print("_______________________________")

'''Data pre-processing and selection'''
print(cell_df.dtypes)

print("_______________________________")

'''It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:
'''

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

print("_______________________________")

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])

print("_______________________________")

'''We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)).'''

y = np.asarray(cell_df['Class'])
print(y[0:5])

print("_______________________________")


'''Train/Test dataset

We split our dataset into train and test set:
'''
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

print("_______________________________")

'''Modeling (SVM with Scikit-learn)

The SVM algorithm offers a choice of kernel functions for performing its processing. 
Basically, mapping data into a higher dimensional space is called kernelling. 
The mathematical function used for the transformation is known as the kernel function,
 and can be of different types, such as:

1.Linear
2.Polynomial
3.Radial basis function (RBF)
4.Sigmoid

Each of these functions has its characteristics, its pros and cons, and its equation, 
but as there's no easy way of knowing which function performs best with any given dataset. 
We usually choose different functions in turn and compare the results. Let's just use the default, 
RBF (Radial Basis Function) for this lab.
'''

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

'''After being fitted, the model can then be used to predict new values:'''
yhat = clf.predict(X_test)
print(yhat[0:5])
print("_______________________________")


'''Evaluation'''

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

'''You can also easily use the f1_score from sklearn library:'''

from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')

'''Let's try the jaccard index for accuracy:
'''
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=2)