import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
import requests


'''Download the  data'''

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
response = requests.get(path)

with open('drug200.csv', 'wb') as my_data:
    my_data.write(response.content)


'''Now, read the data using pandas dataframe'''
my_data = pd.read_csv("drug200.csv", delimiter=",")
print(my_data[0:5])

'''size of data '''
print(my_data.shape)

'''Pre-processing

Using my_data as the Drug.csv data read by pandas, declare the following variables:

    X as the Feature Matrix (data of my_data)
    y as the response vector (target)

Remove the column containing the target name since it doesn't contain numeric values.
'''

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

'''some features in this dataset are categorical, such as Sex or BP.
 Unfortunately, Sklearn Decision Trees does not handle categorical variables. 
 We can still convert these features to numerical values using LabelEncoder to convert the 
 categorical variable into numerical variables.'''

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])

'''Now we can fill the target variable.'''

y = my_data["Drug"]
print(y[0:5])

'''
Setting up the Decision Tree
We will be using train/test split on our decision tree.
 Let's import train_test_split from sklearn.cross_validation.
'''

from sklearn.model_selection import train_test_split

'''Now train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.

The X and y are the arrays required before the split, 
the test_size represents the ratio of the testing dataset, 
and the random_state ensures that we obtain the same splits.'''

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X testing set {}'.format(X_testset.shape),'&',' Size of Y testing set {}'.format(y_testset.shape))

'''Modeling decision tree
We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node. '''

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) # it shows the default parameters

'''Next, 
we will fit the data with the training feature 
matrix X_trainset and training response vector y_trainset '''

drugTree.fit(X_trainset,y_trainset)

'''prediction'''

predTree = drugTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])

'''Evaluation¶
Next, let's import metrics from sklearn and check the accuracy of our model. '''

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

'''Accuracy classification score computes subset accuracy: the set 
of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

In multilabel classification, the function returns the subset accuracy. 
If the entire set of predicted labels for a sample strictly matches with the true set of labels, 
then the subset accuracy is 1.0; otherwise it is 0.0.
'''

'''
Visualization¶

Let's visualize the tree
'''

tree.plot_tree(drugTree)
plt.show()