import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

'''Hyperparameter Tuning in Machine Learning'''

'''Load the Iris data set

X: Features of the Iris dataset (sepal length, sepal width, petal length, petal width).
y: Target labels representing the three species of Iris (setosa, versicolor, virginica).
'''

iris = load_iris()
X = iris.data
y = iris.target

'''Splitting the data into training and test set

test_size=0.2: 20% of the data is used for testing.
random_state=42: Ensures reproducibility of the random split.
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Define the parameter grid: Specify a grid of hyperparameters for the SVM model to search over.
 The grid includes different values for C, gamma, and kernel.
 
 C: Regularization parameter.
gamma: Kernel coefficient.
kernel: Specifies the type of kernel to be used in the algorithm.
 '''
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

'''Initialize the SVC model: Create an instance of the support vector classifier (SVC).'''

svc = SVC()

'''Initialize GridSearchCV: 
Set up the GridSearchCV with the SVC model, the parameter grid, and the desired configuration.

estimator: The model to optimize (SVC).
param_grid: The grid of hyperparameters.
scoring='accuracy': The metric used to evaluate the model's performance.
cv=5: 5-fold cross-validation.
n_jobs=-1: Use all available processors.
verbose=2: Show detailed output during the search.
'''

grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

'''Fit GridSearchCV to the training data: Perform the grid search on the training data.'''
grid_search.fit(X_train, y_train)

'''Check the best parameters and estimator:
 After fitting, print the best parameters and the best estimator found during the grid search.'''
print("Best parameters found: ", grid_search.best_params_)
print("Best estimator: ", grid_search.best_estimator_)

'''Make predictions with the best estimator: Use the best estimator to make predictions on the test set.'''
y_pred = grid_search.best_estimator_.predict(X_test)

'''Evaluate the performance: Evaluate the model's performance on the test set using the classification_report function,
 which provides precision, recall, F1-score, and support for each class.'''
print(classification_report(y_test, y_pred))