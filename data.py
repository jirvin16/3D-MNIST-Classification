import numpy as np

def data_iterator(X, y, batch_size, convolution):
	assert X.shape[0] == y.shape[0]
	index = 0
	while index < X.shape[0]:
		if not convolution:
			yield X[index:index+batch_size,:], y[index:index+batch_size]
		else:
			yield np.expand_dims(X[index:index+batch_size,:], len(X.shape)), y[index:index+batch_size]
		index += batch_size