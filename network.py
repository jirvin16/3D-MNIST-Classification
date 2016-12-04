from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data import data_iterator
import numpy as np
import h5py

import datetime
import os
import time
import sys
from pprint import pprint

def leaky_relu(x, alpha=0.1):
	return tf.maximum(alpha*x, x)


class ConvNN(object):
	def __init__(self, config, sess):

		self.debug = None

		self.sess = sess

		# Training network details
		self.convolution 		   = config.convolution
		self.batch_size            = config.batch_size
		self.epochs                = config.epochs
		self.grad_max_norm 		   = config.grad_max_norm
		self.dropout 			   = config.dropout
		self.optimizer 		  	   = config.optimizer
		self.current_learning_rate = 1 if self.optimizer == "SGD" else 0.001 
		self.loss                  = None
		self.optim 				   = None
		self.logits 			   = None

		self.data_directory 	   = "input/"
		self.is_test 			   = config.mode == 1
		self.validate 			   = config.validate
		
		self.save_every 		   = config.save_every
		self.model_name 		   = config.model_name
		self.model_directory 	   = self.model_name
		self.checkpoint_directory  = os.path.join(self.model_directory, "checkpoints")
		self.log_directory 		   = os.path.join(self.model_directory, "logs")
		
		# Dimensions and initialization parameters
		self.std 				   = 0.01
		self.hidden_dim 	       = config.hidden_dim
		self.embedding_dim 		   = 4096
		self.num_layers 	       = config.num_layers
		self.num_classes		   = 10

		# Model placeholders
		if not self.convolution:
			self.X_batch 	       = tf.placeholder(tf.float32, shape=[None, self.embedding_dim], name="X_batch")
		else:
			self.dim 			   = round(self.embedding_dim ** (1/3))
			self.filter_dim 	   = 5
			self.out_channels 	   = config.out_channels
			self.stride_size 	   = 2
			self.pool_size 		   = 2
			self.pool_stride_size  = 1
			self.X_batch 		   = tf.placeholder(tf.float32, shape=[None, self.dim, self.dim, self.dim, 1], name="X_batch")
		
		self.y_batch 	       	   = tf.placeholder(tf.int32,   shape=None, name="y_batch")
		self.dropout_var 	       = tf.placeholder(tf.float32, name="dropout_var")
		
		if self.is_test:
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
		else:
			self.outfile = os.path.join(self.model_directory, "train.out")

		with open(self.outfile, 'w') as outfile:
			pprint(config.__dict__['__flags'], stream=outfile)
			outfile.flush()

		# Data paths
		if not self.convolution:
			self.mode = "baseline"
		else:
			self.mode = "convolution"
		with h5py.File("input/" + self.mode + "_augmented_data.h5") as hf:
			self.X_train = hf["X_train"][:]
			self.y_train = hf["y_train"][:]
			self.X_test  = hf["X_test"][:]
			self.y_test  = hf["y_test"][:]
			if self.validate:
				self.X_test = hf["X_valid"][:]
				self.y_test = hf["y_valid"][:]


	def build_model(self):
		W_initializer 	 = tf.truncated_normal_initializer(stddev=self.std)
		b_initializer    = tf.constant_initializer(0.1, dtype=tf.float32)

		layer_input = self.X_batch

		if self.convolution:

			# layer = 0
			# use_projection = True

			W_conv  		     = tf.get_variable("W_conv", shape=[self.filter_dim, self.filter_dim, self.filter_dim, 1, self.out_channels],
												   initializer=W_initializer)
			b_conv 		     	 = tf.get_variable("b_conv", shape=self.out_channels,
												   initializer=b_initializer)
			# h_conv			     = tf.nn.relu(tf.nn.conv3d(layer_input, W_conv, 
			h_conv			     = leaky_relu(tf.nn.conv3d(layer_input, W_conv, 
														   strides=[1, self.stride_size, self.stride_size, self.stride_size, 1], padding='SAME') + b_conv)
			h_pool 		     	 = tf.nn.max_pool3d(h_conv, ksize=[1, self.pool_size, self.pool_size, self.pool_size, 1],
												    strides=[1, self.pool_stride_size, self.pool_stride_size, self.pool_stride_size, 1], padding='SAME')
			# conv_output_dim 	 = h_pool.get_shape().as_list()
			conv_output   	     = tf.contrib.layers.flatten(h_pool)
			# conv_output_dim1d 	 = conv_output.get_shape().as_list()[1]
			# residual 			 = tf.contrib.layers.flatten(layer_input)
			# residual_dim 		 = residual.get_shape().as_list()[1]

			# if conv_output_dim1d != self.embedding_dim:
			# 	if use_projection:
			# 		W_proj1 		 = tf.get_variable("W_proj1", shape=[residual_dim, conv_output_dim1d],
			# 											   initializer=W_initializer)
			# 		hidden_input 	 = tf.matmul(residual, W_proj1) + conv_output
			# 	else:
			# 		hidden_input 	 = tf.pad(residual, [[0,0], [0, conv_output_dim1d - self.embedding_dim]], "CONSTANT") + conv_output
			# else:
			# 	hidden_input 		 = residual + conv_output
			hidden_input 		 = conv_output

			# if layer + 1 < self.num_layers:
			# 	layer_input 	 = tf.reshape(hidden_input, [-1] + conv_output_dim[1:])



		else:	
			hidden_input = self.X_batch
		
		hidden_input_dim = hidden_input.get_shape().as_list()[1]

		# Go straight into number of classes if using convolutional network? 
		# ie, move hidden layer into above else block (and move matrix multiply below into else)
		W_hidden  = tf.get_variable("W_hidden", shape=[hidden_input_dim, self.hidden_dim],
											initializer=W_initializer)
		b_hidden  = tf.get_variable("b_hidden", shape=self.hidden_dim,
											initializer=b_initializer)
		W_output  = tf.get_variable("W_output", shape=[self.hidden_dim, self.num_classes],
											initializer=W_initializer)
		b_output  = tf.get_variable("b_output", shape=self.num_classes,
											initializer=b_initializer)

		# self.logits       = tf.matmul(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_input, W_hidden) + b_hidden), self.dropout_var),
		# 							  W_output) + b_output
		# self.logits    = tf.matmul(tf.nn.relu(tf.matmul(hidden_input, W_hidden) + b_hidden),
		self.logits    = tf.matmul(leaky_relu(tf.matmul(hidden_input, W_hidden) + b_hidden),
									  W_output) + b_output

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
				previous_valid_loss = valid_loss
				valid_loss = self.test()

				# if validation loss increases, halt training
				# model in previous epoch will be saved in checkpoint
				# if valid_loss > previous_valid_loss:
				# 	break

			# Adaptive learning rate
			if self.optimizer != "Adagrad" and previous_train_loss <= train_loss + 1e-1:
				self.current_learning_rate /= 2.

			# save model after validation check
			if (epoch % self.save_every == 0 or epoch >= self.epochs / 2) and valid_loss <= best_valid_loss:
				self.saver.save(self.sess,
								os.path.join(self.checkpoint_directory, "MemN2N.model")
								)
				best_valid_loss = valid_loss


	def test(self):

		# only load if in test mode (rather than cv)
		if self.is_test:
			self.load()

		test_loss = 0
		num_batches = 0.0
		num_correct = 0.0
		num_examples = 0.0
		for X_batch, y_batch in data_iterator(self.X_test, self.y_test, self.batch_size):

			feed = {self.X_batch: X_batch, self.y_batch: y_batch, self.dropout_var: 0.0}

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


	def run(self):
		if self.is_test:
			self.test()
		else:
			self.train()

	def load(self):
		with open(self.outfile, 'a') as outfile:
			print(" [*] Reading checkpoints...", file=outfile)
			outfile.flush()
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_directory)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception(" [!] Test mode but no checkpoint found")



