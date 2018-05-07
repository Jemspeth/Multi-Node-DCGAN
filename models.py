import tensorflow as tf
import numpy as np


def leaky_relu(x, alpha=0.2):
	return tf.maximum(x, alpha * x);
'''
def generator(inputs, genNum, isTraining=True, reuse=False):	
		with tf.variable_scope('generator_%d' % genNum, reuse=reuse):
			fully1 = tf.layers.dense(inputs, 1024, name='fully_g')

			#First hidden layer with conv1 of size [100, 4, 4, 512]
			conv1 = tf.layers.conv2d_transpose(fully1, 512, [4, 4], strides=(1, 1),
														  padding='valid', name='conv1_g')
			lrelu1 = leaky_relu(tf.layers.batch_normalization(conv1, training=isTraining))
			print('\nconv1:')
			print(conv1)
			# 2nd hidden layer with conv2 of size [100, 7, 7, 256]
			conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), 
														  padding='valid', name='conv2_g')
			lrelu2 = leaky_relu(tf.layers.batch_normalization(conv2, training=isTraining))
			print('\nconv2:')
			print(conv2)

			# 3rd hidden layer with conv3 of size[100, 14, 14, 128]
			conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2),
														  padding='valid', name='conv3_g')
			lrelu3 = leaky_relu(tf.layers.batch_normalization(conv3, training=isTraining))
			print('\nconv3:')
			print(conv3)

			# 4th hidden layer with conv4 of size [100, 28, 28, 64]
			conv4 = tf.layers.conv2d_transpose(lrelu3, 64, [4, 4], strides=(2, 2),
														  padding='valid', name='conv4_g')
			lrelu4 = leaky_relu(tf.layers.batch_normalization(conv4, training=isTraining))
			print('\nconv4:')
			print(conv4)

			# output layer with conv5 of size [100, 28, 28, 1]
			conv5 = tf.layers.conv2d_transpose(lrelu4, 3, [4,4], strides=(3, 2),
														  padding='same', name='conv5_g')
			gen_output = tf.tanh(conv5, name='output_g')
			print('\nconv5:')
			print(conv5)

			return gen_output

#########################################
#CONSIDER ADDING MAXPOOLING: tf.layers.max_pool(convx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='same')
#########################################
def discriminator(inputs, discrimNum, isTraining=True, reuse=False):
		with tf.variable_scope('discrim_%d' % discrimNum, reuse=reuse):
			#output of conv1 is [batch_size, 69, 46, 64]
			conv1 = tf.layers.conv2d(inputs, 64, [4,4], strides=(2,2),
											 padding='same', name='conv1_d')
			lrelu1 = leaky_relu(conv1)
			print('\nconv1:')
			print(conv1)

			#output of conv2 is [batch_size, 35, 23, 128]
			conv2 = tf.layers.conv2d(lrelu1, 128, [4,4], strides=(2,2),
											 padding='same', name='conv2_d')
			lrelu2 = leaky_relu(tf.layers.batch_normalization(conv2, training=isTraining))
			drop = tf.layers.dropout(lrelu2, rate=0.1)
			print('\nconv2:')
			print(conv2)

			#output of conv3 is [batch_size, 18, 12, 256]
			conv3 = tf.layers.conv2d(drop, 256, [4,4], strides=(2,2),
											 padding='same', name='conv3_d')
			lrelu3 = leaky_relu(tf.layers.batch_normalization(conv3, training=isTraining))
			print('\nconv3:')
			print(conv3)

			#output of conv4 is [batch_size, 4, 4, 512], use no padding
			conv4 = tf.layers.conv2d(lrelu3, 512, [4,4], strides=(2,2),
											 padding='same', name='conv4_d') ####################padding was valid
			lrelu4 = leaky_relu(tf.layers.batch_normalization(conv4, training=isTraining))
			#drop = tf.layers.dropout(lrelu2, rate=0.1)
			print('\nconv4:')
			print(conv4)

			#Flatten to [batch_size, 512]
			flat_logits = tf.contrib.layers.flatten(lrelu4)
			##output is the probability for each image in the batch
			flat_logits = tf.layers.dense(flat_logits, units=1, name='flat_d')
			output = tf.sigmoid(flat_logits, name='output_d')
			print('\nOutput:')
			print(output)

			return flat_logits
'''
#Trying to share all layers except the first dense layer for the generators!!!

def generator(inputs, genNum, isTraining=True, reuse=False):
	with tf.variable_scope('generator_%d' % genNum, reuse=reuse):
		#First hidden layer with conv1 of size [100, 4, 4, 512]
		conv1 = tf.layers.conv2d_transpose(inputs, 512, [4, 4], strides=(1, 1),
													  padding='valid', name='conv1_g')

	with tf.variable_scope('generator_x', reuse=tf.AUTO_REUSE):
		lrelu1 = leaky_relu(tf.layers.batch_normalization(conv1, training=isTraining))
		print('\nconv1:')
		print(conv1)
		# 2nd hidden layer with conv2 of size [100, 7, 7, 256]
		conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), 
													  padding='same', name='conv2_g')
		lrelu2 = leaky_relu(tf.layers.batch_normalization(conv2, training=isTraining))
		print('\nconv2:')
		print(conv2)

		# 3rd hidden layer with conv3 of size[100, 14, 14, 128]
		conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2),
													  padding='same', name='conv3_g')
		lrelu3 = leaky_relu(tf.layers.batch_normalization(conv3, training=isTraining))
		print('\nconv3:')
		print(conv3)

		# 4th hidden layer with conv4 of size [100, 28, 28, 64]
		conv4 = tf.layers.conv2d_transpose(lrelu3, 64, [4, 4], strides=(2, 2),
													  padding='same', name='conv4_g')
		lrelu4 = leaky_relu(tf.layers.batch_normalization(conv4, training=isTraining))
		print('\nconv4:')
		print(conv4)

		# output layer with conv5 of size [100, 28, 28, 1]
		conv5 = tf.layers.conv2d_transpose(lrelu4, 3, [4,4], strides=(2, 2),
													  padding='same', name='conv5_g')
		gen_output = tf.tanh(conv5, name='output_g')
		print('\nconv5:')
		print(conv5)

	return gen_output
	'''
	with tf.variable_scope('generator_%d' % genNum, reuse=reuse):
		fully1 = tf.layers.dense(inputs, 1024, name='fully_g')

	with tf.variable_scope('generator_x', reuse=tf.AUTO_REUSE):
		#First hidden layer with conv1 of size [100, 4, 4, 512]
		conv1 = tf.layers.conv2d_transpose(fully1, 512, [4, 4], strides=(1, 1),
													  padding='valid', name='conv1_g')
		lrelu1 = leaky_relu(tf.layers.batch_normalization(conv1, training=isTraining))
		print('\nconv1:')
		print(conv1)
		# 2nd hidden layer with conv2 of size [100, 7, 7, 256]
		conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), 
													  padding='same', name='conv2_g')
		lrelu2 = leaky_relu(tf.layers.batch_normalization(conv2, training=isTraining))
		print('\nconv2:')
		print(conv2)

		# 3rd hidden layer with conv3 of size[100, 14, 14, 128]
		conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2),
													  padding='same', name='conv3_g')
		lrelu3 = leaky_relu(tf.layers.batch_normalization(conv3, training=isTraining))
		print('\nconv3:')
		print(conv3)

		# 4th hidden layer with conv4 of size [100, 28, 28, 64]
		conv4 = tf.layers.conv2d_transpose(lrelu3, 64, [4, 4], strides=(2, 2),
													  padding='same', name='conv4_g')
		lrelu4 = leaky_relu(tf.layers.batch_normalization(conv4, training=isTraining))
		print('\nconv4:')
		print(conv4)

		# output layer with conv5 of size [100, 28, 28, 1]
		conv5 = tf.layers.conv2d_transpose(lrelu4, 3, [4,4], strides=(2, 2),
													  padding='same', name='conv5_g')
		gen_output = tf.tanh(conv5, name='output_g')
		print('\nconv5:')
		print(conv5)

		return gen_output
'''
'''
		with tf.variable_scope('generator_%d' % genNum, reuse=reuse):
			#First hidden layer with conv1 of size [100, 4, 4, 512]
			conv1 = tf.layers.conv2d_transpose(fully1, 512, [4, 4], strides=(1, 1),
														  padding='valid', name='conv1_g')
			lrelu1 = leaky_relu(tf.layers.batch_normalization(conv1, training=isTraining))
			print('\nconv1:')
			print(conv1)
			# 2nd hidden layer with conv2 of size [100, 7, 7, 256]
			conv2 = tf.layers.conv2d_transpose(lrelu1, 256, [4, 4], strides=(2, 2), 
														  padding='same', name='conv2_g')
			lrelu2 = leaky_relu(tf.layers.batch_normalization(conv2, training=isTraining))
			print('\nconv2:')
			print(conv2)

			# 3rd hidden layer with conv3 of size[100, 14, 14, 128]
			conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [4, 4], strides=(2, 2),
														  padding='same', name='conv3_g')
			lrelu3 = leaky_relu(tf.layers.batch_normalization(conv3, training=isTraining))
			print('\nconv3:')
			print(conv3)

			# 4th hidden layer with conv4 of size [100, 28, 28, 64]
			conv4 = tf.layers.conv2d_transpose(lrelu3, 64, [4, 4], strides=(2, 2),
														  padding='same', name='conv4_g')
			lrelu4 = leaky_relu(tf.layers.batch_normalization(conv4, training=isTraining))
			print('\nconv4:')
			print(conv4)

			# output layer with conv5 of size [100, 28, 28, 1]
			conv5 = tf.layers.conv2d_transpose(lrelu4, 3, [4,4], strides=(2, 2),
														  padding='same', name='conv5_g')
			gen_output = tf.tanh(conv5, name='output_g')
			print('\nconv5:')
			print(conv5)

			return gen_output
'''
#########################################
#CONSIDER ADDING MAXPOOLING: tf.layers.max_pool(convx, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='same')
#########################################
def discriminator(inputs, discrimNum, isTraining=True, reuse=False):
		with tf.variable_scope('discrim_%d' % discrimNum, reuse=reuse):
			#output of conv1 is [batch_size, 69, 46, 64]
			conv1 = tf.layers.conv2d(inputs, 64, [4,4], strides=(2,2),
											 padding='same', name='conv1_d')
			lrelu1 = leaky_relu(conv1)
			print('\nconv1:')
			print(conv1)

			#output of conv2 is [batch_size, 35, 23, 128]
			conv2 = tf.layers.conv2d(lrelu1, 128, [4,4], strides=(2,2),
											 padding='same', name='conv2_d')
			lrelu2 = leaky_relu(tf.layers.batch_normalization(conv2, training=isTraining))
			print('\nconv2:')
			print(conv2)

			#output of conv3 is [batch_size, 18, 12, 256]
			conv3 = tf.layers.conv2d(lrelu2, 256, [4,4], strides=(2,2),
											 padding='same', name='conv3_d')
			lrelu3 = leaky_relu(tf.layers.batch_normalization(conv3, training=isTraining))
			drop = tf.layers.dropout(lrelu2, rate=0.4)
			print('\nconv3:')
			print(conv3)

			#output of conv4 is [batch_size, 4, 4, 512], use no padding
			conv4 = tf.layers.conv2d(lrelu3, 512, [4,4], strides=(2,2),
											 padding='same', name='conv4_d') ####################padding was valid
			lrelu4 = leaky_relu(tf.layers.batch_normalization(conv4, training=isTraining))
			print('\nconv4:')
			print(conv4)

			#Flatten to [batch_size, 512]
			flat_logits = tf.contrib.layers.flatten(lrelu4)
			##output is the probability for each image in the batch
			flat_logits = tf.layers.dense(flat_logits, units=1, name='flat_d')
			output = tf.sigmoid(flat_logits, name='output_d')
			print('\nOutput:')
			print(output)

			return flat_logits
