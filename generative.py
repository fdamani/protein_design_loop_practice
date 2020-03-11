import IPython
import torch

from IPython import embed

def independent_multinomial(pwm, N):
	"""p(y) = p(y_1)p(y_2)p(y_3)p(y_4)...p(y_T)
	
	pwm: num chars x num positions, each column is a multinomial distribution

	N: number of sequences to sample
	"""
	seqs = []
	pwm = torch.tensor(pwm)
	num_positions = pwm.shape[1]
	for _ in range(N):
		seq = []
		for i in range(num_positions):
			seq.append(torch.multinomial(pwm[:, i], 1).item())
		seqs.append(seq)

	return seqs


def rnn_autoregressive(N):
	"""p(y) = p(y_1)p(y_2 | y_1)p(y_3 | y_2, y_1)...p(y_T | y_t<T)

	language model sampler.
	"""
	return None