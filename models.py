import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import sys
import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils


from torch.nn.utils import clip_grad_norm_
from torch import optim

from IPython import embed
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=7
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.

class LinearRegression():
	def __init__(self, model='ridge'):
		self.model = model
		if self.model == 'ridge':
			self.lr = sklearn.linear_model.Ridge(fit_intercept=True)
		else:
			print('error: function only supports ridge regression')
			sys.exit(0)
	def fit(self, X, y):
		self.lr.fit(X, y.flatten())

	def predict(self, X):
		return self.lr.predict(X)

	def get_params(self):
		return self.lr.intercept_, self.lr.coef_

class RandomForest():
	def __init__(self):
		self.lr = sklearn.ensemble.RandomForestRegressor()
	
	def fit(self, X, y):
		self.lr.fit(X, y.flatten())

	def predict(self, X):
		return self.lr.predict(X)

	def get_params(self):
		return self.lr.get_params()

class RegressionRNN(nn.Module):
	'''
	Input: sequence of chars
	Output: regression score for last token.
	'''
	def __init__(self, vocab_size, embed_size, hidden_size):
		super(RegressionRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, input):
		hidden = self.final_layer_embedding(input)
		score = self.score(hidden)
		return score

	def final_layer_embedding(self, input):
		b_size = input.batch_sizes[0].item()
		embedded = nn.utils.rnn.PackedSequence(self.embedding(input.data), 
			input.batch_sizes)
		hidden = self.initHidden(b_size)
		output, hidden = self.gru(embedded, hidden)
		return hidden
	
	def score(self, hidden):
		'''
			fully connected layer outputting a single scalar value
		'''
		return self.fc(hidden)

	def initHidden(self, b_size=1):
		return torch.zeros(1, b_size, self.hidden_size, device=device)

class TrainRNN():
	def __init__(self, proc, 
					   embed_size=56, 
					   hidden_size=128,
					   loss='MSE',
					   opt='Adam',
					   lr=1e-3,
					   batch_size=128,
					   valid_every=1000,
					   max_iter=50000,
					   save_every=2000,
					   save_dir = './rnn_output',
					   vocab_size=20):
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.loss = loss
		self.opt = opt
		self.lr = lr
		self.batch_size = batch_size
		self.valid_every = valid_every
		self.max_iter = max_iter
		self.save_every = save_every
		self.save_dir = save_dir
		self.vocab_size = vocab_size

		self.net = RegressionRNN(self.vocab_size, self.embed_size, self.hidden_size)
		self.net.to(device)
		
		if self.loss == 'MSE':
			self.criterion = nn.MSELoss()
		if self.opt == 'Adam':
			self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

		self.losses, self.avg_loss, self.test_avg_loss, self.r2_avg = [], [], [], []
	
		self.proc = proc

	def train(self, X_train, Y_train, X_test, Y_test):
		# X, Y, vocab_size, n_total_samples, chars_to_int, int_to_chars = process_data(f)
		# X_train, Y_train, X_test, Y_test = train_test_split(X, Y, n_total_samples)
		n_train_samples = len(X_train)
		n_test_samples = len(X_test)

		iter = 0
		n_epochs = int(float(self.max_iter) / (float(n_train_samples) / self.batch_size))
		for epoch in range(n_epochs):
			permutation = torch.randperm(n_train_samples)

			for i in range(0, n_train_samples, self.batch_size):
				self.optimizer.zero_grad()

				# access mini-batch
				indices = permutation[i:i+self.batch_size]
				batch_x, batch_y = self.proc.get_mini_batch(X_train, Y_train, indices)
				batch_packed_x, batch_y, _ = self.proc.get_packed_batch(batch_x, batch_y)

				# forward pass
				output = self.net(batch_packed_x)
				loss = self.criterion(output, batch_y.view(1, -1, 1))
				
				# backprop and gradient step
				loss.backward()
				clip_grad_norm_(self.net.parameters(), 0.5)
				self.optimizer.step()
				with torch.no_grad():
					self.losses.append(loss.item())
					if iter % self.valid_every == 0:
						self.avg_loss.append(np.mean(self.losses))
						print ('iter: ', iter, ' loss: ', np.mean(self.losses),)
						self.losses = []
					if iter % self.valid_every == 0:
						# print validation accuracy
						test_permutation = torch.randperm(n_test_samples)
						valid_loss = []
						valid_preds = []
						valid_target = []
						for j in range(0, n_test_samples, self.batch_size):
							test_indices = test_permutation[j:j+self.batch_size]
							batch_x_test, batch_y_test = self.proc.get_mini_batch(X_test, Y_test, test_indices)
							# how to measure validation error.
							batch_packed_x_test, batch_packed_y_test, _ = self.proc.get_packed_batch(batch_x_test, batch_y_test)
							test_output = self.net(batch_packed_x_test)
							loss = self.criterion(test_output, batch_packed_y_test.view(1, -1, 1))
							valid_loss.append(loss.item())
							valid_preds.append(test_output.flatten())
							valid_target.append(batch_packed_y_test)
						print ('valid: ', np.mean(valid_loss))
						print ('\n')
						self.test_avg_loss.append(np.mean(valid_loss))
						
						plt.cla()
						plt.plot(self.avg_loss, label='train')
						plt.plot(self.test_avg_loss, label='validation')
						plt.legend(loc = 'upper right')
						plt.savefig(self.save_dir+'/loss.png')
						
					if iter % self.save_every == 0 and iter > 0:
						torch.save(self.net, self.save_dir+'/model_'+str(iter)+'.pth')

				iter +=1
	def predict(self, X, net, Y=None):
		"""pass in regular sequences"""
		batchX, batchY, _ = self.proc.get_packed_batch(X, Y)
		preds = net(batchX).squeeze().detach().cpu().numpy()
		if Y is not None:
			return preds, batchY.cpu().numpy()
		else:
			return preds