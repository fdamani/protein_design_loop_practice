import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import processing
import models
import evaluation
import dim_reduction as dr
import pwm
import generative as gen

import torch
import numpy as np
import IPython

import scipy
from scipy import stats

from processing import Processing
from models import LinearRegression, RandomForest, TrainRNN
from evaluation import Evaluation
from IPython import embed


def neural_regression(proc, data, trainRNN=False, model_path=None):
	print('neural regression...')

	trainIndsX, trainY, testIndsX, testY = data

	# train RNN
	rnn = TrainRNN(proc=proc)
	if trainRNN:
		rnn.train(trainIndsX, trainY, testIndsX, testY)
		net = rnn.net
	else:
		if torch.cuda.is_available():
			net = torch.load(model_path)
		else:
			net = torch.load(model_path, map_location=torch.device('cpu'))

	# inference
	batchTrainPreds, batchTrainY = rnn.predict(trainIndsX, net, trainY)
	print('train: ', metrics.r2_score(batchTrainY, batchTrainPreds))
	batchTestPreds, batchTestY = rnn.predict(testIndsX, net, testY)
	print('valid: ', metrics.r2_score(batchTestY, batchTestPreds))

	return net, rnn

def one_hot_regression(proc, data):
	print('one hot regression...')
	"""ridge and random forest regression"""
	# train and test 
	trainOneHotX, trainY, testOneHotX, testY = data

	######### linear regression #########
	linear_models = 'ridge'
	print('ridge regression with one hot representation...')
	linReg = LinearRegression(model=linear_models)	
	linReg.fit(trainOneHotX, trainY)
	preds = linReg.predict(testOneHotX)
	print('test r2 score: ', metrics.r2_score(testY, preds))
	print('test mse: ', metrics.mse(testY, preds))

	return linReg
	
	# ######### random forest #########
	# print('random forest...')
	# rf = RandomForest()
	# rf.fit(trainOneHotX, trainY)
	# preds = rf.predict(testOneHotX)
	# print('test r2 score: ', metrics.r2_score(testY, preds))
	# print('test mse: ', metrics.mse(testY, preds))


def pca(data, net):
	"""compute pca plots for one hot and neural embedding.
		- compare variance explained
		- dimensionality reduction
	"""
	trainOneHotX, testOneHotX, testX, trainY, testY = data

	neuralEmbedding = proc.sequences_to_neural_embedding(testX, net)

	neuralPCA = dr.pca(neuralEmbedding, 20)
	oneHotPCA = dr.pca(testOneHotX, 20)
	
	# plot explained variance ratio
	plt.cla()
	plt.plot(neuralPCA.explained_variance_ratio_, label='neural')
	plt.plot(oneHotPCA.explained_variance_ratio_, label='one hot')
	plt.xlabel('num pca components')
	plt.ylabel('explained variance ratio')
	plt.legend(loc = 'upper right')
	plt.savefig('rnn_output/representation_pca_variance_explained.png')

	# plot neural pca in 2d
	neuralPCA_2d = dr.pca_fit_transform(neuralEmbedding, 2)
	plt.cla()
	plt.scatter(x=neuralPCA_2d[:, 0], y=neuralPCA_2d[:, 1], c=testY.flatten(), label='neural')
	plt.xlabel('pca 1')
	plt.ylabel('pca 2')
	plt.savefig('rnn_output/neural_2d_pca.png')

	# plot one hot pca in 2d
	oneHotPCA_2d = dr.pca_fit_transform(trainOneHotX, 2)
	plt.cla()
	plt.scatter(oneHotPCA_2d[:, 0], oneHotPCA_2d[:, 1], c=trainY.flatten(), label='one hot')
	plt.legend(loc = 'upper right')
	plt.savefig('rnn_output/one_hot_2d_pca.png')

	return oneHotPCA_2d

def pwms(proc, trainX, trainY, y_thresh=3.5, top=True):
	"""compute position weight matrices"""
	if top:
		x = trainX[trainY > y_thresh]
	else:
		x = trainX[trainY < y_thresh]
	return pwm.compute_counts(x, proc)


def gen_model_independent_mult(pwm_topy, topY, proc, linReg, net, rnn):
	# independent multinomials with pwm computed on top y scoring compounds
	numCompounds = 1000
	rand_inds = np.random.permutation(np.arange(len(topY)))
	topY = topY[rand_inds[0:numCompounds]]

	gen_inds = gen.independent_multinomial(pwm_topy, numCompounds)
	gen_seqs = proc.inds_matrix_to_seqs(gen_inds)
	novel_seqs, percent_novel = metrics.filter_to_novel(gen_seqs, proc.seq)
	novel_inds = proc.seq_matrix_to_inds(novel_seqs)
	print('percent novel: ', percent_novel)

	neuralPreds  = rnn.predict(novel_inds, net)
	linRegPreds = linReg.predict(proc.sequences_to_one_hot(novel_seqs))

	metrics.score_histogram(neuralPreds, topY, 'rnn_output', '/hist_independent_mult_neuralPreds.png')
	metrics.score_histogram(linRegPreds, topY, 'rnn_output', '/hist_independent_mult_linRegPreds.png')


	# # in general, the linear regressor's predictions don't go higher than 3
	# lx = linReg.predict(testOneHotX)
	# print(lx[lx > 3])
	# print(len(testY[testY>3]))

	return neuralPreds, linRegPreds, novel_seqs

if __name__ == "__main__":
	########### initial processing #########
	csv_file = 'challenge.txt'
	# read in data
	proc = Processing(csv_file)

	# train / test split
	trainX, testX, trainY, testY = proc.train_test_split(proc.seq, proc.y)

	# compute one hot embedding
	trainOneHotX, testOneHotX = proc.sequences_to_one_hot(trainX), proc.sequences_to_one_hot(testX)

	# compute seq to inds
	trainIndsX, testIndsX = proc.seq_matrix_to_inds(trainX), proc.seq_matrix_to_inds(testX)
	
	# initialize metrics object
	metrics = Evaluation()

	############################ regression ############################ 

	# neural regression
	net, rnn = neural_regression(proc=proc,
		              data=(trainIndsX, trainY, testIndsX, testY),
		              trainRNN=False,
		              model_path='rnn_output/model_6000iter_validlosspt250.pth')

	# # one hot regression
	linReg = one_hot_regression(proc=proc,
					   data=(trainOneHotX, trainY, testOneHotX, testY))


	############################ PCA on one hot and neural embedding ############################
	oneHotPCA_2d = pca(data=(trainOneHotX, testOneHotX, testX, trainY, testY), net=net)


	############################ generative modeling ############################
	
	# generative model of the entire data distribution
	y_thresh = -100.0 # pwm for the entire dataset
	pwm_topy = pwms(proc, trainX, trainY, y_thresh=y_thresh, top=True)
	topY = trainY[trainY > y_thresh]

	# independent multinomials with pwm computed on top y scoring compounds
	neuralPredsAll, linRegPredsAll, genSeqsAll = gen_model_independent_mult(pwm_topy, topY, proc, linReg, net, rnn)

	print('neural prediction mean: ', np.mean(neuralPredsAll), ' std: ', np.std(neuralPredsAll))
	embed()

	# generative model of sequences with a score greater than 3.5
	y_thresh = 3.5 # pwm for the entire dataset
	pwm_topy = pwms(proc, trainX, trainY, y_thresh=y_thresh, top=True)
	topY = trainY[trainY > y_thresh]

	neuralPredsTop, linRegPredsTop, genSeqsTop = gen_model_independent_mult(pwm_topy, topY, proc, linReg, net, rnn)
	print('neural prediction mean: ', np.mean(neuralPredsTop), ' std: ', np.std(neuralPredsTop))

	embed()

	y_thresh = -.5 # pwm for the entire dataset
	pwm_bottomy = pwms(proc, trainX, trainY, y_thresh=y_thresh, top=False)
	bottomY = trainY[trainY < y_thresh]

	neuralPredsBottom, linRegPredsBottom, genSeqsBottom = gen_model_independent_mult(pwm_bottomy, bottomY, proc, linReg, net, rnn)
	print('neural prediction mean: ', np.mean(neuralPredsBottom), ' std: ', np.std(neuralPredsBottom))

	embed()
