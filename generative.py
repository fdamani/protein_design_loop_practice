import IPython
import torch
import numpy as np
from IPython import embed

class IndependentMult:
	"""fully factorized generative model where each dimension is independent
	p(x) = p(x_1)p(x_2)p(x_3)p(x_4)...p(x_T)
	"""
	
	def __init__(self):
		pass

	def estimate(self, X, proc):
		"""estimate parameters of multinomial distributions

		Compute PWMs: normalized frequency of observing each character
		for each position. Add pseudocounts (5) to each entry.
		
		:X sequences
		:proc processing object

		return params num chars x num positions
		"""
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		indsX = proc.seq_matrix_to_inds(X)
		numChars = indsX.shape[1]
		N = indsX.shape[0]
		pseudoCounts = 5
		pwm_mat = np.zeros((20, numChars))
		# compute normalized frequency of observing each character for each position
		for i in range(numChars):
			freqs = np.zeros(20)
			observed_counts = np.bincount(indsX[:, i])
			freqs[0: len(observed_counts)] = observed_counts
			pwm_mat[:, i] = freqs + pseudoCounts # add pseudocounts
			pwm_mat[:, i] = pwm_mat[:, i] / np.sum(pwm_mat[: ,i])
		return pwm_mat

	def sample(self, params, N, proc=None):
		"""p(x) = p(x_1)p(x_2)p(x_3)p(x_4)...p(x_T)
			
		:params num chars x num positions, each col is a mult distribution
		:N number of samples

		return list of N sequences

		"""
		seqs = []
		params = torch.tensor(params)
		num_positions = params.shape[1]
		for _ in range(N):
			seq = []
			for i in range(num_positions):
				seq.append(torch.multinomial(params[:, i], 1).item())
			seqs.append(seq)

		return seqs

class Ar1Shared:
	"""1st order AR model with shared parameters across time-steps
	p(x) = p(x_1; \phi)p(x_2|x_1; \theta)p(x_3|x_2; \theta)...p(x_T|x_{T-1}; \theta)
	"""
	def __init__(self):
		pass

	def estimate(self, X, proc):
		"""estimate frequencies of transition matrix and prior
		:X sequences N x 1
		:proc processing object

		return
			:prior parameters to a multinomial distribution of length n_chars p(x1)
			:transition_mat n_chars x n_chars, where each row is the conditional p(x_t|x_{t-1}=i)
		"""
		indsX = proc.seq_matrix_to_inds(X)
		
		# compute prior
		observed_counts = np.bincount(indsX[:, 0])
		prior = np.zeros(20)
		prior[0: len(observed_counts)] = observed_counts
		prior = prior + 5
		prior = prior / np.sum(prior)

		transition_mat = np.zeros((20, 20))
		for i in range(len(indsX)):
			for j in range(0, len(indsX[i]) - 1):
				row_val = indsX[i][j]
				col_val = indsX[i][j+1]
				transition_mat[row_val][col_val] += 1

		# add pseudocounts then normalize rows
		for i in range(len(transition_mat)):
			transition_mat[i] = transition_mat[i] + 5
			transition_mat[i] = transition_mat[i] / np.sum(transition_mat[i])

		params = (prior, transition_mat)
		return params

	def sample(self, model_params, N, proc):
		"""Sample from AR1 Shared model
		:model_params tuple (prior, transition_mat)
		:N number of samples
		:proc processing object

		return N sampled sequences
		"""
		prior, transition_mat = model_params
		seqs = []
		for _ in range(N):
			seq = []
			seq.append(torch.multinomial(torch.tensor(prior), 1).item())
			for i in range(proc.seq_len - 1):
				seq.append(torch.multinomial(torch.tensor(transition_mat[seq[-1]]), 1).item())
			seqs.append(seq)
		return seqs

class Ar1NoShare:
	"""1st order AR model with parameters per time-step
	p(x) = p(x_1; \phi)p(x_2|x_1; \theta_1)p(x_3|x_2; \theta_2)...p(x_T|x_{T-1}; \theta_{T-1})
	"""
	def __init__(self):
		pass

	def estimate(self, X, proc):
		"""Estimate frequencies of seq_len-1 transition matrices and prior
		:X sequences N x 1
		:proc processing object
		return
			:prior parameters to a multinomial distribution of length n_chars p(x1)
			:transition_mats seq_len-1 x n_chars x n_chars, where first dimension
				indexes a transition matrix between t-1 and t. 
		"""
		indsX = proc.seq_matrix_to_inds(X)
		# compute prior
		observed_counts = np.bincount(indsX[:, 0])
		prior = np.zeros(20)
		prior[0: len(observed_counts)] = observed_counts
		prior = prior + 5
		prior = prior / np.sum(prior)

		transition_mats = []
		for i in range(proc.seq_len-1):
			transition_mats.append(np.zeros((20, 20)))

		# loop over samples
		for i in range(len(indsX)):
			# loop over positions
			for j in range(0, proc.seq_len - 1):
				row_val = indsX[i][j]
				col_val = indsX[i][j+1]
				# position-specific parameter estimation
				transition_mats[j][row_val][col_val] += 1
		# add pseudocounts then normalize rows
		for j in range(0, proc.seq_len-1):
			for i in range(len(transition_mats[j])):
				transition_mats[j][i] = transition_mats[j][i] + 5
				transition_mats[j][i] = transition_mats[j][i] / np.sum(transition_mats[j][i])
		
		params = (prior, transition_mats)
		return params

	def sample(self, model_params, N, proc):
		"""Sample from AR1 No Share model
		:model_params tuple (prior, transition_mats)
		:N number of samples
		:proc processing object

		return N sampled sequences
		"""
		prior, transition_mats = model_params
		seqs = []
		for _ in range(N):
			seq = []
			seq.append(torch.multinomial(torch.tensor(prior), 1).item())
			for i in range(proc.seq_len - 1):
				seq.append(torch.multinomial(torch.tensor(transition_mats[i][seq[-1]]), 1).item())
			seqs.append(seq)
		return seqs



















