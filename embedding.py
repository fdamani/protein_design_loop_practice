import sklearn
import processing
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from processing import Processing
import IPython
from IPython import embed
class Embedding:
	def __init__(self):
		pass
	
	def seq_matrix_to_inds(self, X):
		"""convert matrix of sequences to matrix of indices"""
		N = len(X)
		indsX = np.array([self.seq_to_ind(list(X[i][0])) for i in range(N)])
		return indsX		

	def inds_matrix_to_one_hot(self, X):
		"""given a matrix of indices, convert to a matrix of one hots"""
		return np.array([self.ind_to_one_hot(X[i]) for i in range(len(X))])

	def sequences_to_one_hot(self, X):
		"""given matrix of sequences, convert to one hots"""
		indsX = self.seq_matrix_to_inds(X)
		oneHotsX = self.inds_matrix_to_one_hot(indsX)
		return oneHotsX
