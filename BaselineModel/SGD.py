# Template code retrieved from : https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
# Other resource : https://realpython.com/gradient-descent-algorithm-python/
# Then modified to fit our project
from Skeleton_Dataset.load_skeleton_tracking_ucla import *
from load_skeletondata_DKE import *
import numpy as np

class SGD: # STOCHASTIC GRADIENT DESCENT USING RANDOM MINI-BATCHES
    def __init__(self, lr=0.01, max_iter=1000, batch_size=32, tol=1e-3):
        self.learning_rate = lr # Small learning rates can result in very slow convergence.
        self.max_iteration = max_iter
        self.batch_size = batch_size
        self.tolerence_convergence = tol # Tolerance for stopping criterion.
        self.theta = None

    def fit(self, X, y):
        n, _, _ = X.shape
        X_flat = X.reshape(n, -1)  # Flatten the last dimensions
        _, d = X_flat.shape
        y = y.reshape(-1, 1) # Reshape y into a column vector 
        self.theta = np.random.randn(d)
        self.theta = self.theta.reshape(-1, 1)  # Reshape self.theta to be the same shape as y

        for _ in range(self.max_iteration):
            indices = np.random.permutation(n)
            X_flat = X_flat[indices]
            y = y[indices]

            for i in range(0, n, self.batch_size):
                X_batch = X_flat[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                grad = self.gradient(X_batch, y_batch)
                self.theta -= self.learning_rate * grad  # Update theta/weights

            if np.linalg.norm(self.learning_rate * grad) < self.tolerence_convergence: # if the vector/weight update in the current iteration is less than or equal to tolerance stop iterating.
                break

    def gradient(self, X, y): # gradient is used in fit, no need to flatten X
        n = len(y)
        y_pred = np.dot(X, self.theta)
        error = y_pred - y
        grad = np.dot(X.T, error) / n
        return grad

    def predict(self, X):
        X_flat = X.reshape(len(X), -1) # Flatten the last dimensions
        y_pred = np.dot(X_flat, self.theta)
        return y_pred
    
    def load_weights(self, filename="SGD_model_weights.npy"):
        self.theta = np.load(filename)
        
    def save_weights(self, filename="SGD_model_weights.npy"):
        np.save(filename, self.theta)
        
## SGD MODEL INITIALIZATION ##
def initialize():
    # Load & Preprocess data
    dsamp_train, dsamp_test, tr_fea_xyz, tr_label, tr_seq_len, te_fea_xyz, te_label, te_seq_len = preprocess_ucla("BaselineModel/Skeleton_Dataset/ucla_data")
    # print ("no error in loading + preprocessing data")

    # Make sure len(tr_fea_xyz) = len(tr_label) 

    # Create random dataset with 100 rows and 5 columns
    X = np.array(tr_fea_xyz)
    # print("no error in X declaration") 

    # create corresponding target value by adding random noise in the dataset
    # random noise avoids getting stuck in local minima
    y = np.dot(X.T, np.array(tr_label)) + np.random.randn(60, 50, 10) * 0.1
    # print("no error in y declaration")

    # Create an instance of the SGD class
    model = SGD(lr=0.01, max_iter=1000, batch_size=32, tol=1e-3) # change param if needed
    # print("no error in model declaration")

    model.fit(X, y) 

    # Predict using predict method from model
    y_pred = model.predict(X) # Matrix of predicted values
    print(y_pred) 

    # Save model parameters & weights
    # model.save_weights("Project3-1/BaselineModel/SGD_model_weights.npy")
    print("model weights saved")

## MAIN PROGRAM ## (Run only once to initialize model)
if __name__ == "__main__":
    initialize() # Only run this once in the file, weights will be saved in "SGD_model_weights.npy"





