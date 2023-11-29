# Template code retrieved from : https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
# Other resource : https://realpython.com/gradient-descent-algorithm-python/
# Then modified to fit our project
import random
from Skeleton_Dataset.load_datasetSGD import *
import numpy as np

class SGD:
    def __init__(self, lr=0.01, max_iter=500, batch_size=32, tol=1e-3):
        self.learning_rate = lr
        self.max_iteration = max_iter
        self.batch_size = batch_size
        self.tolerance_convergence = tol
        self.theta = None

    def fit(self, X, y):
        # store dimensions of input tensor
        n, d, v = X.shape
        # Initialize random Theta for every feature
        self.theta = np.random.randn(d, v) # change theta
        print(self.theta.shape)
        for _ in range(self.max_iteration):
            # Shuffle the data
            indices = random.randint(0, n-(self.batch_size+1))
            X_batch = X[indices:indices+self.batch_size] # Select 32 skeletons starting from random index between 0 and 214
            y_batch = y[indices:indices+self.batch_size] # Select 32 skeletons starting from random index between 0 and 214
            
            # Iterate over mini-batches of keypoints
            for i in range(0, self.batch_size):
                X_skel = X_batch[i] # Select a skeleton amongst the random batch
                y_skel = y_batch[i] # Select a skeleton amongst the random batch
                grad = self.gradient(X_skel, y_skel) # Perform gradient descent on two random skeletons
                self.theta -= self.learning_rate * grad 
            # Check for convergence
            if np.linalg.norm(grad) < self.tolerance_convergence:
                break

    def gradient(self, X, y):
        n = len(y)
        y_pred = X * self.theta  # theta has to have the same dimensions as X with X being a random skeleton
        error = y_pred - y
        grad = (X * error) / n  # element-wise multiplication and sum along the first axis
        return grad

    def predict(self, X):
        y_pred = X * self.theta  # element-wise multiplication
        return y_pred
    
    def load_weights(self, filename="SGD_model_weights.npy"):
        self.theta = np.load(filename)
        
    def save_weights(self, filename="SGD_model_weights.npy"):
        np.save(filename, self.theta)
        
## SGD MODEL INITIALIZATION ##
def initialize():
    # Load & Preprocess data
    data = getdata()
    print ("no error in loading + preprocessing data")

    X = np.array(data)
    skeleton_features = np.array(data)
    print("no error in X and features declaration") 
    
    # create corresponding target value by adding random noise in the dataset
    # random noise avoids getting stuck in local minima
    y = (X * skeleton_features) + np.random.randn(247, 34, 3) * 0.1 
    print("no error in y declaration")

    # Create an instance of the SGD class
    model = SGD(lr=0.01, max_iter=1000, batch_size=32, tol=1e-3) # change param if needed
    # print("no error in model declaration")

    model.fit(X, y) 
    print("no error in model fitting")

    # Predict using predict method from model
    # X is number of skeletons right now I'm predicting the whole training data so 247 skeletons in advance
    # But I can predict one or n skeletons at a time by changing the dimensions of X
    # X doesn't necessarily have to be the training data
    # y_pred = model.predict(X) 
    # print("no error in prediction")

    # Save model parameters & weights
    model.save_weights("BaselineModel/SGD_model_weights.npy")
    print("model weights saved")

## MAIN PROGRAM ## (Run only once to initialize model)
if __name__ == "__main__":
    initialize() # Only run this once in the file, weights will be saved in "SGD_model_weights.npy"





