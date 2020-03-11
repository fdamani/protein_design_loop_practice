Design loop:

Using the sequences and scores, train some models that predict score given a sequence.

Try a few different models of varying complexity and compare the performance.

Using the sequences alone, learn a generative function (maybe a few strategies as time allows). Use your favorite model from part 1 as a surrogate, score generated sequences and compare.

(Might be trickier) Generate ~20 sequences in different score buckets (e.g. very low score vs very high score)

Bonus points for generating PWMs for the buckets

=======================

Framework:

1. Processing file X
	- reads in csv returns x y
	- standardizes y values
	- train/test split (use random seed to ensure reproducability)
2. Embedding file
	- one hot representation X
	- neural embedding
	- pca
3. Regression models file
	- linear regressor
		- fit
		- predict
		- return coefficients
	- eval class
		- measure error in predictions on held out data.
4. Generative models
	- language model for generation
	- independent multinomial distribution -> sample
		- given a dataset, compute pwm matrix with random being background then sample
	- evaluate by compute KS statistic on generated data comparing predicted values for generated things compared
		to actual training data.
5. optimization
	- involves generative model and regression model
	- generate things, score, iterative training, then repeat.

===========

things i tried that didn't help:

- linear regression on 2d pca
- visualizing neural embeddings
	- 2d pca
	- 2d tsne


- how to choose top points for pwm?
	- pca dim 1
	- just take the compounds with the highest score