import h5py
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

with h5py.File("input/baseline_augmented_data.h5") as hf:
	X_train = hf["X_train"][:]
	y_train = hf["y_train"][:]
	X_valid = hf["X_valid"][:]
	y_valid = hf["y_valid"][:]
	X_test  = hf["X_test"][:]
	y_test  = hf["y_test"][:]

# print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

def create_linear_svm(params):
	if 'penalty' in params:
		penalty = params['penalty']
	else:
		penalty = 'l2'
	if 'loss' in params:
		loss = params['loss']
	else:
		loss = 'squared_hinge'
	if 'dual' in params:
		dual = params['dual']
	else:
		dual = True
	return LinearSVC(penalty=penalty, loss=loss, dual=dual)

def create_svm(params):
	if 'decision_function_shape' in params:
		decision_function_shape = params['decision_function_shape']
	else:
		decision_function_shape = None
	if 'kernel' in params:
		kernel = params['kernel']
	else:
		kernel = 'rbf'
	return SVC(decision_function_shape=decision_function_shape, kernel=kernel)

def create_logistic_regression(params):
	if 'multi_class' in params:
		multi_class = params['multi_class']
	else:
		multi_class = 'ovr'
	if 'solver' in params:
		solver = params['solver']
	else:
		solver = 'liblinear'
	return LogisticRegression(multi_class=multi_class, solver=solver)

# linear_params = [{}, {'penalty':'l1', 'dual': False}, {'loss': 'hinge'}]

# print("Linear SVM")
# for param in linear_params:
# 	print(param)
# 	clf = create_linear_svm(param)
# 	clf.fit(X_train, y_train)
# 	print("Valid Score: ", clf.score(X_valid, y_valid))

# params = [{}, {'decision_function_shape':'ovo'}, {'decision_function_shape':'ovr'},
# 		  {'decision_function_shape':'ovo', 'kernel': 'poly'},
# 		  {'decision_function_shape':'ovo', 'kernel': 'sigmoid'},
# 		  {'decision_function_shape':'ovr', 'kernel': 'poly'},
# 		  {'decision_function_shape':'ovr', 'kernel': 'sigmoid'}]

# print("Nonlinear SVM")
# for param in params:
# 	print(param)
# 	clf = create_svm(param)
# 	clf.fit(X_train, y_train)
# 	print("Valid Score: ", clf.score(X_valid, y_valid))

print("Logistic Regression")
log_params = [{}, {'multi_class':'multinomial', 'solver':'newton-cg'}]
for param in log_params:
	print(param)
	clf = create_logistic_regression(param)
	clf.fit(X_train, y_train)
	print("Valid Score: ", clf.score(X_valid, y_valid))




