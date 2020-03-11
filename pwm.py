"""create position weight matrices"""
import IPython
import numpy as np
from IPython import embed

def compute_counts(X, proc):
	"""Compute PWMs: normalized frequency of observing each character
	for each position. Add pseudocounts (5) to each entry.
	
	X is sequences
	proc is processing object

	return PWM num chars x num positions
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