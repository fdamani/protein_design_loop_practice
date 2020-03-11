import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
class Evaluation:
	def __init__(self):
		pass

	def r2_score(self, true, preds):
		return sklearn.metrics.r2_score(true, preds)

	def mse(self, true, preds):
		return sklearn.metrics.mean_squared_error(true, preds)


	def filter_to_novel(self, gen, true):
		"""compute percent of samples in gen not in true
		gen, true (str): sequences
		return percent novel
		"""
		N = len(gen)
		hits = 0
		novel_samples = []
		for i in range(N):
			if gen[i] in true:
				hits += 1
			else:
				novel_samples.append(gen[i])
		return np.array(novel_samples).reshape(-1,1), (N - hits) / N

	def compute_score(self, gen, score):
		N = len(gen)
		scores = []
		return np.array([scores.append(score(gen[i])) for i in range(N)])


	def score_histogram(self, gen_scores, data_scores, save_dir, file_name):
		"""plot histogram comparing the scores from gen compounds
		and training data"""
		plt.cla()
		plt.hist(gen_scores, label='gen scores', alpha=.5,bins=20)
		plt.hist(data_scores, label='training data scores', alpha=.5,bins=20)
		plt.legend(loc='upper left')
		plt.savefig(save_dir+file_name)