from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from data import data_iterator
import numpy as np
import h5py

import datetime
import os
import time
import sys
from pprint import pprint
from collections import Counter
import plot_confusion_matrix
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.1):
	return tf.maximum(alpha*x, x)


class ConvNN(object):
	def __init__(self, config, sess):

		self.debug = None

		self.sess = sess

		random_seed  			   = config.random_seed
		np.random.seed(random_seed)
		tf.set_random_seed(random_seed)

		# Training network details
		self.convolution 		   = config.convolution
		self.batch_size            = config.batch_size
		self.epochs                = config.epochs
		self.grad_max_norm 		   = config.grad_max_norm
		self.dropout 			   = config.dropout
		self.optimizer 		  	   = config.optimizer
		self.current_learning_rate = 1 if self.optimizer == "SGD" else 0.001 
		self.hidden_nonlinearity   = config.hidden_nonlinearity.lower()
		assert self.hidden_nonlinearity in ["leaky_relu", "relu", "sigmoid", "tanh", "elu", "none"]
		self.loss                  = None
		self.optim 				   = None
		self.logits 			   = None
		self.activations 		   = None

		self.data_directory 	   = "../data/"
		self.voxel_dim  		   = config.voxel_dim
		self.is_test 			   = config.mode == 1
		self.visuals			   = config.mode == 2
		self.validate 			   = config.validate
		
		self.save_every 		   = config.save_every
		self.model_name 		   = config.model_name
		self.model_directory 	   = self.model_name
		self.checkpoint_directory  = os.path.join(self.model_directory, "checkpoints")
		self.log_directory 		   = os.path.join(self.model_directory, "logs")
		
		# Dimensions and initialization parameters
		self.std 				   = 0.01
		self.hidden_dim 	       = config.hidden_dim
		self.embedding_dim 		   = self.voxel_dim ** 3
		self.num_layers 	       = config.num_layers
		self.num_classes		   = 10
		self.filter_dim 	   	   = 5
		self.out_channels 	   	   = config.out_channels
		self.stride_size 	   	   = 2
		self.pool_size 		   	   = 2
		self.pool_stride_size  	   = 1

		# Model placeholders
		if not self.convolution:
			self.X_batch 	       = tf.placeholder(tf.float32, shape=[None, self.embedding_dim], name="X_batch")
		else:
			self.X_batch 		   = tf.placeholder(tf.float32, shape=[None, self.voxel_dim, self.voxel_dim, self.voxel_dim, 1], name="X_batch")
		
		self.y_batch 	       	   = tf.placeholder(tf.int32,   shape=None, name="y_batch")
		self.dropout_var 	       = tf.placeholder(tf.float32, name="dropout_var")
		
		if self.is_test or self.visuals:
			self.dropout = 0

		if not os.path.isdir(self.data_directory):
			raise Exception(" [!] Data directory %s not found" % self.data_directory)

		if not os.path.isdir(self.model_directory):
			os.makedirs(self.model_directory)

		if not os.path.isdir(self.log_directory):
			os.makedirs(self.log_directory)

		if not os.path.isdir(self.checkpoint_directory):
			if self.is_test:
				raise Exception(" [!] Checkpoints directory %s not found" % self.checkpoint_directory)
			else:
				os.makedirs(self.checkpoint_directory)

		if self.is_test:
			self.outfile = os.path.join(self.model_directory, "test.out")
		elif self.visuals:
			self.outfile = os.path.join(self.model_directory, "rotations.out")
		else:
			self.outfile = os.path.join(self.model_directory, "train.out")

		with open(self.outfile, 'w') as outfile:
			pprint(config.__dict__['__flags'], stream=outfile)
			outfile.flush()

		# Data
		with h5py.File(os.path.join(self.data_directory, "data.h5")) as hf:
			self.X_train = hf["X_train"][:]
			self.y_train = hf["y_train"][:]
			self.X_test  = hf["X_test"][:]
			self.y_test  = hf["y_test"][:]
			if self.validate:
				self.X_test = hf["X_valid"][:]
				self.y_test = hf["y_valid"][:]
			if self.convolution:
				self.X_train = np.expand_dims(self.X_train.reshape(self.X_train.shape[0], self.voxel_dim, self.voxel_dim, self.voxel_dim), 4)
				self.X_test  = np.expand_dims(self.X_test.reshape(self.X_test.shape[0], self.voxel_dim, self.voxel_dim, self.voxel_dim), 4)


	def build_model(self):
		W_initializer 	 = tf.truncated_normal_initializer(stddev=self.std)
		b_initializer    = tf.constant_initializer(0.1, dtype=tf.float32)

		layer_input = self.X_batch

		if self.convolution:

			W_conv  		     = tf.get_variable("W_conv", shape=[self.filter_dim, self.filter_dim, self.filter_dim, 1, self.out_channels],
												   initializer=W_initializer)
			b_conv 		     	 = tf.get_variable("b_conv", shape=self.out_channels,
												   initializer=b_initializer)
			h_conv			     = leaky_relu(tf.nn.conv3d(layer_input, W_conv, 
														   strides=[1, self.stride_size, self.stride_size, self.stride_size, 1], padding='SAME') + b_conv)
			if self.dropout > 0:
				h_conv = tf.nn.dropout(h_conv, 1-self.dropout_var)

			if self.num_layers > 1:
				W_conv2  		     = tf.get_variable("W_conv2", shape=[3, 3, 3, self.out_channels, self.out_channels],
												   		initializer=W_initializer)
				b_conv2 		     = tf.get_variable("b_conv2", shape=self.out_channels,
												   		initializer=b_initializer)
				h_conv 				 = leaky_relu(tf.nn.conv3d(h_conv, W_conv2,
															   strides=[1, 1, 1, 1, 1], padding='SAME') + b_conv2)
				
			h_pool 		     	 = tf.nn.max_pool3d(h_conv, ksize=[1, self.pool_size, self.pool_size, self.pool_size, 1],
												    strides=[1, self.pool_stride_size, self.pool_stride_size, self.pool_stride_size, 1], padding='SAME')
			if self.dropout > 0:
				h_pool = tf.nn.dropout(h_pool, 1-self.dropout_var)
			
			conv_output   	     = tf.contrib.layers.flatten(h_pool)

			hidden_input 		 = conv_output

		else:	
			hidden_input = self.X_batch
		
		hidden_input_dim = hidden_input.get_shape().as_list()[1]

		W_hidden  = tf.get_variable("W_hidden", shape=[hidden_input_dim, self.hidden_dim],
									initializer=W_initializer)
		b_hidden  = tf.get_variable("b_hidden", shape=self.hidden_dim,
									initializer=b_initializer)
		W_output  = tf.get_variable("W_output", shape=[self.hidden_dim, self.num_classes],
									initializer=W_initializer)
		b_output  = tf.get_variable("b_output", shape=self.num_classes,
									initializer=b_initializer)
		hidden_output = tf.matmul(hidden_input, W_hidden) + b_hidden
		
		if self.hidden_nonlinearity == "leaky_relu":
			hidden_output = leaky_relu(hidden_output)
		elif self.hidden_nonlinearity == "relu":
			hidden_output = tf.nn.relu(hidden_output)
		elif self.hidden_nonlinearity == "sigmoid":
			hidden_output = tf.sigmoid(hidden_output)
		elif self.hidden_nonlinearity == "tanh":
			hidden_output = tf.tanh(hidden_output)
		elif self.hidden_nonlinearity == "elu":
			hidden_output = tf.nn.elu(hidden_output)

		self.activations = hidden_output

		if self.dropout > 0:
			hidden_output = tf.nn.dropout(hidden_output, 1-self.dropout_var)

		self.logits    = tf.matmul(hidden_output, W_output) + b_output

		batch_loss 	   = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.y_batch)
		self.loss 	   = tf.reduce_mean(batch_loss)

		# only optimize if training
		if not self.is_test:
			self.optim = tf.contrib.layers.optimize_loss(self.loss, None, self.current_learning_rate, self.optimizer, clip_gradients=self.grad_max_norm, 
														 summaries=["learning_rate", "gradient_norm", "loss", "gradients"])
		
		self.sess.run(tf.initialize_all_variables())

		with open(self.outfile, "a") as outfile:
			for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				print(var.name, file=outfile)
				print(var.get_shape(), file=outfile)	
				outfile.flush()

		self.saver = tf.train.Saver()

	def train(self):

		total_loss = 0.0

		merged_sum = tf.merge_all_summaries()
		t 	   	   = datetime.datetime.now()
		writer     = tf.train.SummaryWriter(os.path.join(self.log_directory, \
										    "{}-{}-{}-{}-{}-{}".format(t.year, t.month, t.day, t.hour % 12, t.minute, t.second)), \
										    self.sess.graph)

		i 					= 0
		previous_train_loss = float("inf")
		valid_loss 			= float("inf")
		best_valid_loss     = float("inf")
		tolerance 			= 0
		for epoch in xrange(self.epochs):

			train_loss  = 0.0
			num_batches = 0.0
			for X_batch, y_batch in data_iterator(self.X_train, self.y_train, self.batch_size):

				feed = {self.X_batch: X_batch, self.y_batch: y_batch, self.dropout_var: self.dropout}

				_, batch_loss, summary = self.sess.run([self.optim, self.loss, merged_sum], feed)

				train_loss += batch_loss
				
				if i % 10 == 0:
					writer.add_summary(summary, i)
				
				i += 1
				num_batches += 1.0

			state = {
				"train_loss" : train_loss / num_batches,
				"epoch" : epoch,
				"learning_rate" : self.current_learning_rate,
			}

			with open(self.outfile, 'a') as outfile:
				print(state, file=outfile)
				outfile.flush()
			
			if self.validate:
				valid_loss = self.test()

				# if validation loss increases, halt training
				# model in previous epoch will be saved in checkpoint
				if valid_loss > best_valid_loss:
					if tolerance >= 10:
						break
					else:
						tolerance += 1
				# save model after validation check
				else:
					tolerance = 0
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)
					best_valid_loss = valid_loss
			
			else:
				if epoch % self.save_every == 0:
					self.saver.save(self.sess,
									os.path.join(self.checkpoint_directory, "MemN2N.model")
									)

			# Adaptive learning rate
			if self.optimizer != "Adagrad" and previous_train_loss <= train_loss / num_batches + 1e-1:
				self.current_learning_rate /= 2.

			previous_train_loss = train_loss / num_batches


	def test(self):

		# only load if in test mode (rather than cv)
		if self.is_test:
			self.load()

		test_loss = 0
		num_batches = 0.0
		num_correct = 0.0
		num_examples = 0.0
		for X_batch, y_batch in data_iterator(self.X_test, self.y_test, self.batch_size):

			feed = {self.X_batch: X_batch, self.y_batch: y_batch, self.dropout_var: self.dropout}

			loss, logits = self.sess.run([self.loss, self.logits], feed)

			test_loss += loss

			predictions = np.argmax(logits, 1)
			num_correct += np.sum(predictions == y_batch)
			num_examples += predictions.shape[0]

			num_batches += 1.0

		state = {
			"test_loss" : test_loss / num_batches,
			"accuracy" : num_correct / num_examples
		}

		with open(self.outfile, 'a') as outfile:
			print(state, file=outfile)
			outfile.flush()

		return test_loss / num_batches

	def visualize(self):

		self.load()

		all_predictions = np.array([])
		all_labels      = np.array([])
		all_activations = np.array([]).reshape(0, self.hidden_dim)

		for X_batch, y_batch in data_iterator(self.X_test, self.y_test, self.batch_size):

			feed = {self.X_batch: X_batch, self.y_batch: y_batch, self.dropout_var: self.dropout}

			logits, activations = self.sess.run([self.logits, self.activations], feed)

			all_activations = np.concatenate((all_activations, activations))

			predictions = np.argmax(logits, 1)

			all_predictions = np.concatenate((all_predictions, predictions))
			all_labels      = np.concatenate((all_labels, y_batch))

		best_positive_neurons = np.argpartition(sum(all_activations[1::2] == 1), -5)[-5:]
		best_negative_neurons = np.argpartition(sum(all_activations[1::2] < 0.00001), -5)[-5:]

		cm = confusion_matrix(all_predictions, all_labels)

		plot_confusion_matrix.plot_confusion_matrix(cm, [str(i) for i in range(10)], self.model_directory)

		with h5py.File(os.path.join(self.data_directory, "rotations.h5")) as h5f:
			samples = h5f["X_train"][:]
			samples = np.expand_dims(samples.reshape(samples.shape[0], self.voxel_dim, self.voxel_dim, self.voxel_dim), 4)
			labels = h5f["y_train"][:]

		for X_batch, y_batch in data_iterator(samples, labels, samples.shape[0]):

			feed = {self.X_batch: X_batch, self.y_batch: y_batch, self.dropout_var: self.dropout}

			activations, = self.sess.run([self.activations], feed)

		positive = activations[:, best_positive_neurons]
		negative = activations[:, best_negative_neurons]

		plt.matshow(positive)
		plt.title("Top 5 Positive Neuron Activations")
		plt.xlabel("Neuron")
		plt.ylabel("Type of Rotation")
		plt.savefig("positive.png")
		plt.close()
		plt.matshow(negative)
		plt.title("Top 5 Negative Neuron Activations")
		plt.xlabel("Neuron")
		plt.ylabel("Type of Rotation")
		plt.savefig("negative.png")
		plt.close()

	def run(self):
		if self.is_test:
			self.test()
		elif self.visuals:
			self.visualize()
		else:
			self.train()

	def load(self):
		with open(self.outfile, 'a') as outfile:
			print(" [*] Reading checkpoints...", file=outfile)
			outfile.flush()
		print(self.checkpoint_directory)
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_directory)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception(" [!] Test mode but no checkpoint found")



