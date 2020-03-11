"""perform pca and dim reduction"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import sklearn
import numpy as np
from sklearn import decomposition, manifold

import IPython

from IPython import embed

def pca(X, n_components=100):
	"""standardize data, perform pca and return pca object"""
	# X = (X - np.mean(X, axis=0) / np.std(X, axis=0))
	pcaObj = sklearn.decomposition.PCA(n_components=n_components).fit(X)
	return pcaObj

def pca_fit_transform(X, n_components=2):
	"""standardize data, perform pca, and return data projected into pca space"""
	# X = (X - np.mean(X, axis=0) / np.std(X, axis=0))
	return sklearn.decomposition.PCA(n_components=n_components).fit_transform(X)

def tsne_fit_transform(X, n_components=2):
	X_emb = sklearn.manifold.TSNE(n_components=n_components).fit_transform(X)
	return X_emb