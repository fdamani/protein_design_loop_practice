import pandas as pd
import numpy as np
import torch
import IPython
import torch
import torch.nn as nn
from IPython import embed

class Processing:

	def __init__(self, csv_file):

		self.chars = ['I', 'A', 'N', 'L', 'S', 'P', 
			  'H', 'W', 'V', 'C', 'M', 'G', 
			  'T', 'F', 'R', 'K', 'Q', 'Y', 'E', 'D']

		self.csv_file = csv_file
		# return seq and y in csv file
		self.seq, self.y = self.process(csv_file)
		# standardize y values
		self.y = self.standardize(self.y)
		self.seq_len = 12
	
		#### test cases #####
		testSeq = 'QTI'
		assert self.seq_to_ind(testSeq) == [16, 12, 0]
		oneHots = np.zeros(60)
		oneHots[16] = 1
		oneHots[20+12] = 1
		oneHots[20+20+0] = 1
		assert (self.ind_to_one_hot(self.seq_to_ind(testSeq)) == oneHots).all()
		assert 'QTI' == self.ind_to_seq([16, 12, 0])

	def process(self, file):
		"""read csv file and return seq and value columns"""
		df = pd.read_table(file)
		seq, y = df['seq'].values[:, None], df['value'].values[:, None]
		return seq, y

	def standardize(self, y):
		"""standardize y values to have mean 0 and sd 1"""
		return (y - np.mean(y)) / np.std(y)

	def train_test_split(self, x, y, percentTrain = .8):
		"""train test split"""
		# set random seed for reproducability
		np.random.seed(7)
		N = len(x)
		permutedInds = np.random.permutation(np.arange(N))
		trainInds = permutedInds[0 : int(percentTrain * N)]
		testInds = permutedInds[int(percentTrain * N) :]
		return x[trainInds], x[testInds], y[trainInds], y[testInds]

	def seq_to_ind(self, x):
		"""given a string x, convert to indices"""
		inds = []
		numChars = len(list(x))
		for i in range(numChars):
			inds.append(self.chars.index(x[i]))
		return inds
	
	def ind_to_seq(self, x):
		"""given a vector of indices, convert to string"""
		seq = []
		for i in range(len(x)):
			seq.append(self.chars[x[i]])
		return ''.join(seq)

	def ind_to_one_hot(self, x):
		"""given a vector x, convert to a flattened vector of one hots

			x is a vector
			convert to a matrix of one hots
			return flattened matrix
		"""
		return np.eye(len(self.chars))[x].flatten()

	def seq_matrix_to_inds(self, X):
		"""convert matrix of sequences to matrix of indices"""
		if len(X.shape)==1:
			X = X.reshape(-1,1)
		N = len(X)
		indsX = np.array([self.seq_to_ind(list(X[i][0])) for i in range(N)])
		return indsX		

	def inds_matrix_to_seqs(self, X):
		"""convert matrix of indices to a matrix of sequences"""
		N = len(X)
		seqs = np.array([self.ind_to_seq(X[i]) for i in range(N)])
		return seqs

	def inds_matrix_to_one_hot(self, X):
		"""given a matrix of indices, convert to a matrix of one hots"""
		return np.array([self.ind_to_one_hot(X[i]) for i in range(len(X))])

	def sequences_to_one_hot(self, X):
		"""given matrix of sequences, convert to one hots"""
		indsX = self.seq_matrix_to_inds(X)
		oneHotsX = self.inds_matrix_to_one_hot(indsX)
		return oneHotsX

	def sequences_to_neural_embedding(self, X, net):
		"""given matrix of sequences and a model, convert to neural embeddings"""
		# convert to inds
		indsX = self.seq_matrix_to_inds(X)
		# convert to format readable by rnn 
		batchX, _, _ = self.get_packed_batch(indsX, None)
		# compute embeddings
		emb = net.final_layer_embedding(batchX).squeeze().detach().cpu().numpy()
		return emb
	
	def get_mini_batch(self, X, Y, indices):
		# get minibatch from rand indices

		batch_x = [X[ind] for ind in indices]
		if Y is None:
			return batch_x, None
		else:
			batch_y = Y[indices]
			return batch_x, batch_y

	def get_packed_batch(self, batch_x, batch_y):
		# get length of each sample in mb
		batch_lens = [len(sx) for sx in batch_x]
		# arg sort batch lengths in descending order
		sorted_inds = np.argsort(batch_lens)[::-1]
		batch_x = [batch_x[sx] for sx in sorted_inds]
		batch_packed_x = nn.utils.rnn.pack_sequence([torch.LongTensor(s) for s in batch_x])
		if torch.cuda.is_available():
			batch_packed_x = batch_packed_x.to('cuda')

		if batch_y is None:
			return batch_packed_x, None, sorted_inds
		else:
			batch_y = torch.stack([torch.FloatTensor(batch_y[sx]) for sx in sorted_inds])
			if torch.cuda.is_available():
				batch_y = batch_y.to('cuda')
			return batch_packed_x, batch_y, sorted_inds
