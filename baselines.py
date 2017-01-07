import h5py
import os
import time

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

# for voxel_dim in [8, 16, 32]:
for voxel_dim in [8, 16]:
	
	with h5py.File("input/final_data_{}.h5".format(voxel_dim)) as hf:
		X_train = hf["X_train"][:]
		y_train = hf["y_train"][:]
		X_valid = hf["X_valid"][:]
		y_valid = hf["y_valid"][:]
		X_test  = hf["X_test"][:]
		y_test  = hf["y_test"][:]

	print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

	with open("score_baselines_{}.txt".format(voxel_dim), 'w') as outfile:

		linear_params = [{}, {'penalty':'l1', 'dual': False}]

		print("Linear SVM", file=outfile)
		for param in linear_params:
			print(param, file=outfile)
			clf = create_linear_svm(param)
			clf.fit(X_train, y_train)
			print("Test Score: ", clf.score(X_test, y_test), file=outfile)

		params = [{}, {'decision_function_shape':'ovr', 'kernel': 'poly'}, {'decision_function_shape':'ovr', 'kernel': 'sigmoid'}]

		print("Nonlinear SVM", file=outfile)
		for param in params:
			print(param, file=outfile)
			clf = create_svm(param)
			clf.fit(X_train, y_train)
			print("Test Score: ", clf.score(X_test, y_test), file=outfile)

		print("Logistic Regression", file=outfile)
		log_params = [{}, {'multi_class':'multinomial', 'solver':'newton-cg'}]
		for param in log_params:
			print(param, file=outfile)
			clf = create_logistic_regression(param)
			clf.fit(X_train, y_train)
			print("Test Score: ", clf.score(X_test, y_test), file=outfile)




