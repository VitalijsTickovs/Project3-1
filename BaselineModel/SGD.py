# Template code retrieved from : https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/

import numpy as np

# TODO : Create a vectorizing function for the SGD Class to vectorize data ?

class SGD:
	def __init__(self, lr=0.01, max_iter=1000, batch_size=32, tol=1e-3):
		# learning rate of the SGD Optimizer
		self.learning_rate = lr 
		# maximum number of iterations for SGD Optimizer
		self.max_iteration = max_iter 
		# mini-batch size of the data 
        # The SGD is not strict since batch_sizes aren't one --> this reduces noise and allows for faster computations.
		self.batch_size = batch_size 
		# tolerance for convergence for the theta 
		self.tolerence_convergence = tol 
		# Initialize model parameters to None
		self.theta = None
		
	def fit(self, X, y):
		# store dimension of input vector 
		n, d1, d2 = X.shape
		# print("n, d1, d2 = ", n, d1, d2)
		
		# Intialize random Theta for every feature 
		self.theta = np.random.randn(d1*d2)
		
		for i in range(self.max_iteration):
			# Shuffle the data
			indices = np.random.permutation(n)
			X = X[indices]
			y = y[indices]
			
			# Iterate over mini-batches (We are taking mini-batches of size 32)
			for i in range(0, n, self.batch_size):
				X_batch = X[i:i+self.batch_size]
				y_batch = y[i:i+self.batch_size]
				grad = self.gradient(X_batch, y_batch)
				
				# gradient descent update
				self.theta -= self.learning_rate * grad
				
			# Check for convergence
			if np.linalg.norm(grad) < self.tolerence_convergence:
				break
	
	# define a gradient function for calculating gradient of the data 
	def gradient(self, X, y):
		n = len(y) 
		# predict target value by taking dot product of dependent and theta value 
		y_pred = np.dot(X, self.theta)
		
		# calculate error between predict and actual value 
		error = y_pred - y
		grad = np.dot(X.T, error) / n
		return grad
	
	def predict(self, X):
		# predict y value using calculated theta value 
		y_pred = np.dot(X, self.theta)
		return y_pred

# ### Testing the SGD class ###

# # Create random dataset with 100 rows and 5 columns
# X = np.random.randn(100, 5) # Change this

# # create corresponding target value by adding random noise in the dataset
# y = np.dot(X, np.array([1, 2, 3, 4, 5])) + np.random.randn(100) * 0.1 # change first part
# # np array needs to have the same number of columns as X

# # Create an instance of the SGD class
# model = SGD(lr=0.01, max_iter=1000, #change param if needed
# 			batch_size=32, tol=1e-3)
# model.fit(X, y)

# # Predict using predict method from model
# y_pred = model.predict(X) # Matrix of predicted values




