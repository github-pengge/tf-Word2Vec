# -*- coding: utf-8 -*-
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
# from matplotlib import pylab as plt
from sklearn.manifold import TSNE
import pickle


def read_data(filename):
	"""Extract the first file enclosed in a zip file as a list of words"""
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

filename = 'dataset/text8.zip'
words = read_data(filename)
print('Data size %d' % len(words))


vocabulary_size = 50000

def build_dataset(words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count = unk_count + 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


data_index = 0
def generate_batch(batch_size, half_window_size):
	global data_index
	batch = np.ndarray(shape=(batch_size, 2*half_window_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	len_data = len(data)
	for i in range(batch_size):
		index = data_index
		labels[i] = data[(index+half_window_size)%len_data]
		for k in range(2*half_window_size+1):
			if k != half_window_size:
				t = (k if k < half_window_size else k-1)
				batch[i, t] = data[(index+k)%len_data]
		data_index = (data_index + 1) % len_data
	return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for half_window_size in [1, 2]:
	data_index = 0
	batch, labels = generate_batch(batch_size=8, half_window_size=half_window_size)
	print('\nwith half_window_size = %d:' % (half_window_size))
	print('    batch:', [[reverse_dictionary[b] for b in bi] for bi in batch])
	print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
half_window_size = 1 # How many words to consider left and right.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.int32, shape=(batch_size, 2*half_window_size))
	tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
	tf_valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	embeddings = tf.Variable(tf.random_uniform(shape=(vocabulary_size, embedding_size), minval=-1.0, maxval=1.0))
	softmax_weights = tf.Variable(tf.truncated_normal(shape=(vocabulary_size, embedding_size), stddev=1.0 / math.sqrt(embedding_size)))
	softmax_biases = tf.constant(np.zeros(shape=(vocabulary_size), dtype=np.float32))

	embed = tf.nn.embedding_lookup(embeddings, tf_train_dataset)
	inputs = tf.reduce_sum(embed, 1)
	loss = tf.reduce_mean(
						tf.nn.sampled_softmax_loss(
								softmax_weights, softmax_biases, inputs, tf_train_labels, num_sampled, vocabulary_size
							)
						)

	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

	valid_embed = tf.nn.embedding_lookup(embeddings, tf_valid_dataset)
	similarity = tf.matmul(valid_embed, tf.transpose(softmax_weights)) + softmax_biases
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	norm_ = tf.sqrt(tf.reduce_sum(tf.square(softmax_weights), 1, keep_dims=True))
	normalized_softmax_weights = softmax_weights / norm_
	norm_ = tf.sqrt(tf.reduce_sum(tf.square(normalized_softmax_weights+normalized_embeddings), 1, keep_dims=True))
	normalized_embeddings_2 = (normalized_softmax_weights+normalized_embeddings) / 2.0 / norm_


num_steps = 100001
with tf.Session(graph=graph) as session:
	if int(tf.VERSION.split('.')[1]) > 11:
		tf.global_variables_initializer().run()
	else:
		tf.initialize_all_variables().run()
	print('Initialized')

	average_loss = 0.0
	for step in range(num_steps):
		train_batch, train_labels = generate_batch(batch_size, half_window_size)
		feed_dict = {tf_train_dataset: train_batch, tf_train_labels: train_labels}
		l, _ = session.run([loss, optimizer], feed_dict=feed_dict)
		average_loss += l

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000.0
			print('Average loss at step %d: %f' % (step, average_loss))
			average_loss = 0
		if step % 10000 == 0:
			sim = similarity.eval()
			for i in range(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8 # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[1:top_k+1]  # let alone itself, so begin with 1
				log = 'Nearest to %s:' % valid_word
				for k in range(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log = '%s %s,' % (log, close_word)
				print(log)

	final_embeddings = normalized_embeddings.eval()
	final_embeddings_2 = normalized_embeddings_2.eval()  # this is better

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
two_d_embeddings_2 = tsne.fit_transform(final_embeddings_2[1:num_points+1, :])

with open('2d_embedding_cbow.pkl', 'wb') as f:
	pickle.dump([two_d_embeddings, two_d_embeddings_2, reverse_dictionary], f)
