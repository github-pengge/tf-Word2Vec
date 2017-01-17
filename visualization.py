# -*- coding: utf-8 -*-
from matplotlib import pylab as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages


def plot(embeddings, labels, save_to_pdf='embed.pdf'):
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	pp = PdfPages(save_to_pdf)
	plt.figure(figsize=(15,15))  # in inches
	for i, label in enumerate(labels):
		x, y = embeddings[i,:]
		plt.scatter(x, y)
		plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
						ha='right', va='bottom')
	plt.savefig(pp, format='pdf')
	plt.show()
	pp.close()

method = 'skip_gram'
filename = '2d_embedding_%s.pkl' % method
with open(filename, 'rb') as f:
	[two_d_embeddings, two_d_embeddings_2, reverse_dictionary] = pickle.load(f)

num_points = len(two_d_embeddings)
words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words, save_to_pdf='two_d_embeddings_%s.pdf' % method)
plot(two_d_embeddings_2, words, save_to_pdf='two_d_embeddings_2_%s.pdf' % method)
