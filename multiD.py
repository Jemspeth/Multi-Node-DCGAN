import os, time, imageio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data
from models import generator, discriminator
'''
discrim_num = 1
generator_num = 1
train = True;
batch_size = 40
train_size = 11381
train_epoch = 200
input_height = 138
input_width = 92
col_dim = 3
y_dim = None
z_dim = 100
discrim_lr = 0.00015
gen_lr = 0.0002
smooth = 0.05
data_path = '../movieData/'
model_path = './movieSavedModels/'
image_path = './movieSampleImages/'
plot_path = './movieLossPlots/'
log_path = './movieLogs/'
'''
discrim_num = 2
generator_num = 2
train = True;
batch_size = 128
train_size = 202599
train_epoch = 20
input_height = 64
input_width = 64
col_dim = 3
y_dim = None
z_dim = 100
discrim_lr = 0.00015
gen_lr = 0.0002
smooth = 0.0
data_path = '../faceDataResized/'
model_path = './faceSavedModels/'
image_path = './faceSampleImages/'
plot_path = './faceLossPlots/'
log_path = './faceLogs/'
trial_log = './trialLogs/'
summary_path = 'faceSummaries/'
trial = 4

def main(_):
	train_images = load_dataset(data_path)
	fixed_z = np.random.normal(0, 1, [batch_size, 1, 1, z_dim])

	z = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, z_dim))
	x = tf.placeholder(tf.float32, shape=(batch_size, input_height, input_width, col_dim))

	train_bool = tf.placeholder(dtype=tf.bool)

	#build networks
	g_x = [generator(z, g_num) for g_num in range(generator_num)]
	sampler = [generator(z, g_num, isTraining=False, reuse=True) for g_num in range(generator_num)]
			

	d_real_logits = [discriminator(x, num) for num in range(discrim_num)]
	d_fake_logits = [discriminator(sampler[g_num], num, reuse=True) for g_num in range(generator_num)
											 for num in range(discrim_num)]
	g_train_logits = [discriminator(g_x[g_num], num, reuse=True) for g_num in range(generator_num)
											  for num in range(discrim_num)]


	#calculate losses
	d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
							 logits=d_real_logits, labels=tf.ones_like(d_real_logits) * (1.0 - smooth)))
	d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
							 logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
	discrim_loss = d_real_loss + d_fake_loss

	gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
						 logits=g_train_logits, labels=tf.ones_like(g_train_logits)))

	tf.summary.scalar('Discriminator Loss', discrim_loss)
	tf.summary.scalar('Generator Loss', gen_loss)

	#extract trainable variables for g and d
	var = tf.trainable_variables()
	d_vars = [[v for v in var if v.name.startswith('discrim_%d' % num)] for num in range(discrim_num)]
	g_vars = [[v for v in var if v.name.startswith('generator_%d' % num)] for num in range(generator_num)]
	g_vars.append([v for v in var if ('generator_x') in v.name])


	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		discrim_optim = tf.train.AdamOptimizer(learning_rate=discrim_lr, beta1=0.5) \
										.minimize(discrim_loss, var_list=d_vars)

		gen_optim = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5) \
								  .minimize(gen_loss, var_list=g_vars)


	gpu_options = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

	merged_summary = tf.summary.merge_all()

	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	writer = tf.summary.FileWriter(summary_path + str(trial), sess.graph)

	for v in tf.global_variables():
		print(v.name)


	if( train ):
		np.random.shuffle(train_images)
		print('Training images:')
		print(type(train_images))
		print(train_images.shape)

		train_images = train_images.astype('float32')
		#train_images = train_images/127.5 - 1
		#train_images = (train_images - 127.5) / 127.5
		train_images = train_images/np.max(train_images)
		print('Range: ', np.ptp(train_images))
		print('Min: ', np.amin(train_images))
		print('Max: ', np.amax(train_images))
		print('Mean: ', np.mean(train_images))

		randGen = np.random.randint(0, generator_num, 9)
		print(randGen)

		train_iter = train_size // batch_size
		train_begin = time.time()
		d_loss_arr = [None] * train_epoch * train_iter
		g_loss_arr = [None] * train_epoch * train_iter
		index = 0

		for epoch in range(train_epoch):
			start_time = time.time()
			print('Epoch: %d' % epoch)
			z_gen = np.random.normal(0, 1, [batch_size, 1, 1, z_dim])

			for counter in range(train_iter):
				x_batch = train_images[counter*batch_size :
									 (counter+1)*batch_size]

				z_batch = np.random.normal(0, 1, [batch_size, 1, 1, z_dim])

				_, d_curr_loss = sess.run([discrim_optim, discrim_loss],
									  feed_dict={x: x_batch, z: z_batch})
				d_loss_arr[index] = d_curr_loss

				_, g_curr_loss = sess.run([gen_optim, gen_loss],
									 feed_dict={z: z_batch, x: x_batch})
				g_loss_arr[index] = g_curr_loss

				z_batch = np.random.normal(0, 1, [batch_size, 1, 1, z_dim])
				_, g_curr_loss = sess.run([gen_optim, gen_loss],
									 feed_dict={z: z_batch, x: x_batch})

				index += 1

				if counter % 200 == 0:
					print('\tStep %d: Discriminator loss: %.3f   Generator loss: %.3f' 
					% (counter, np.mean(d_loss_arr), np.mean(g_loss_arr)))

				if counter % 500 == 0:
					#sample_images(epoch, sess)
					samples = sess.run(sampler, feed_dict={z: fixed_z})
					fig, axes = plt.subplots(3, 3)
					fig.suptitle('Epoch %d_%d' % (epoch, counter/500))

					for i in range(9):
						img = np.reshape(samples[randGen[i]][i*9], (input_height, input_width, col_dim))
						axes[i/3, i % 3].imshow(img)

					fig.savefig(image_path + 'trial%d/%d_%d.png' % (trial, epoch, counter/500))
					plt.close(fig)

			saver.save(sess, model_path, global_step=epoch)
			print('Discriminator loss: %.3f   Generator loss: %.3f' 
					% (np.mean(d_loss_arr), np.mean(g_loss_arr)))

			end_time = time.time()
			print('Epoch %d took %.2f' % (epoch, end_time - start_time))

		train_end = time.time()
		total_time = train_end - train_begin;
		print('Total training time: %.2f' % total_time)
		log_trial(total_time)
		plot_hist(d_loss_arr, g_loss_arr)
		make_gif()
		#os.system('systemctl poweroff') #For long training sessions


def sample_images(epoch, sess):
	samples = sess.run(sampler, feed_dict={z: fixed_z})
	fig, axes = plt.subplots(3, 3)
	fig.suptitle('Epoch %d' % epoch)

	for i in range(9):
		img = np.reshape(samples[i * 9], (input_height, input_width, col_dim))
		axes[i/3, i % 3].imshow(img)

	fig.savefig(image_path + '%d_%d.png' % (trial, epoch))
	plt.close(fig)

def make_gif():
	img_arr = []
	for i in range(train_epoch):
		im = imageio.imread(image_path + 'trial%d/%d_%d.png' % (trial, i, 1))
		img_arr.append(im)
	imageio.mimsave(image_path + '%d_animation.gif' % trial, img_arr, fps=3)

def plot_hist(d_loss, g_loss):
	y1 = np.arange(0.0, len(d_loss), 10.0)
	y2 = np.arange(0.0, len(g_loss), 10.0)
	plt.plot(y1, d_loss[0::10], 'r-', label='Discriminator Loss')
	plt.plot(y2, g_loss[0::10], 'b-', label='Generator Loss')
	plt.legend()
	plt.ylabel('Total Loss')
	plt.xlabel('Iterations (10\'s)')
	plt.savefig(plot_path + '%d_trainPlot.png' % trial)

def load_dataset(dir_path):
	print('Loading data from: %s' % dir_path)
	images = []
	for file in os.listdir(dir_path):
		img = mpimg.imread(os.path.join(dir_path, file))
		images.append(img)
	images = np.asarray(images)

	print(images.shape)
	return images

def log_trial(total_time):
	with open(trial_log + str(trial) + '.txt', 'w') as outfile:
		outfile.write('Discrim Count: %d\n' % discrim_num)
		outfile.write('Generator Count: %d\n' % generator_num)
		outfile.write('D and G Learning Rates: %f %f\n' % (discrim_lr, gen_lr))
		outfile.write('Batch Size: %d\n' % batch_size)
		outfile.write('Epochs: %d\n' % train_epoch)
		outfile.write('Smooth: %f\n' % smooth)
		outfile.write('Dropout: Yes\n')
		outfile.write('Total Time: %f\n' % total_time)
		outfile.write('Shared Generator Input Dense Layer: NO\n')



if __name__ == '__main__':
	tf.app.run()
