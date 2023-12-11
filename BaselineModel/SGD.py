# Template code retrieved from : https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
# Other resource : https://realpython.com/gradient-descent-algorithm-python/
# Then modified to fit our project
import random
from Skeleton_Dataset.load_datasetSGD import *
from Skeleton_Dataset.normalize_datasetSGD import *
import numpy as np
import torch as tor
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as Math


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
        # print(self.theta.shape)
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
    
    
    def evaluate_loss(self, y_pred, X):
        random_index = random.randint(0, len(X)-2)
        prediction = y_pred[random_index]
        skeleton_label = X[random_index+1]

        loss = nn.L1Loss()
        cross_loss = nn.CrossEntropyLoss()

        skeleton_label = tor.tensor(skeleton_label)
        prediction = tor.tensor(prediction)

        l1_loss = loss(skeleton_label, prediction)
        cross_entropy_loss = cross_loss(skeleton_label, prediction)
        
        return (l1_loss/34), (cross_entropy_loss/34)
    
    
    def avrg_distance(self, y_pred, X):
        distances = []
        for i in range(len(y_pred)-1):
            prediction = y_pred[i]
            target = X[i+1]
            # Calculate euclidean distance between prediction and target
            euclidean_distance = np.sqrt(np.sum((prediction - target)**2))/(34*3)
            distances.append(euclidean_distance)

        # Get Average of distances
        avrg = np.mean(distances)
        return avrg
    

    def convert_2_cm(self, avrg_distance, Skeleton): 
        keypoint_2 = Skeleton[1][1]
        keypoint_26 = Skeleton[25][1]

        # Calculate euclidean distance between keypoint 2 axis y and 26 axis y 
        euclidean_distance = Math.sqrt((keypoint_2 - keypoint_26)**2)

        # Convert euclidean distance to cm
        average_distance_cm = (avrg_distance * 19.9)/euclidean_distance

        return average_distance_cm
       
        
    def plot(self,Skeleton, Prediction_Skeleton):
        # Create a 3D axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Scatter plot in 3D
        x,y,z = [],[],[]
        for i in range(len(Skeleton)):
            x.append(Skeleton[i][0])
            y.append(Skeleton[i][1])
            z.append(Skeleton[i][2]) # For 3D plot

        ax.scatter(x, y, c='r', marker='o', label='Target Plot')

        x_pred,y_pred,z_pred = [],[],[]
        for i in range(len(Prediction_Skeleton)):
            x_pred.append(Prediction_Skeleton[i][0])
            y_pred.append(Prediction_Skeleton[i][1])
            z_pred.append(Prediction_Skeleton[i][2]) # For 3D plot

        print("absolute distance: ", abs(Skeleton[0][0] - Prediction_Skeleton[0][0]))

        ax.scatter(x_pred, y_pred, c='r', marker='x', label='Prediction Plot')

        # Adding labels and title
        ax.set_xlabel('X-axis Label')
        ax.set_ylabel('Y-axis Label')
        # ax.set_zlabel('Z-axis Label')
        ax.set_title('2D Scatter Plot')

        # Adding a legend
        ax.legend()

        # Display the plot
        plt.show()


    def load_weights(self, filename="SGD_model_weights.npy"):
        self.theta = np.load(filename)
        

    def save_weights(self, filename="SGD_model_weights.npy"):
        np.save(filename, self.theta)
        


## SGD MODEL INITIALIZATION ##
def initialize():
    # Load & Preprocess data
    data = getdata()
    data = normalize_data(data)
    # print ("no error in loading + preprocessing data")

    # Split data into training and testing 
    X_train = []
    X_test = []
    split = 0.7*len(data) # 70% train, 30% test
    for i in range(len(data)):
        if i <= split:
            X_train.append(data[i])
        else:
            X_test.append(data[i])
   
    X = np.array(X_train)
    skeleton_features = X
    # print("no error in X and features declaration") 
    
    # create corresponding target value by adding random noise in the dataset
    # random noise avoids getting stuck in local minima
    y = (X * skeleton_features) + np.random.randn(X.shape[0], X.shape[1], X.shape[2]) * 0.1 
    # print("no error in y declaration")

    # Create an instance of the SGD class
    model = SGD(lr=0.001, max_iter=1000, batch_size=32, tol=1e-3) # change param if needed
    # print("no error in model declaration")

    model.fit(X, y) 
    # print("no error in model fitting")

    # Predict using predict method from model
    # the predict parameter influences the length of the prediction in the future
    y_pred = model.predict(X_test) 
    # print(y_pred)

    # Evaluate loss
    l1_loss,cross_entropy_loss = model.evaluate_loss(y_pred, X_test)
    print("l1_loss:", l1_loss)
    print("cross_entropy_loss:", cross_entropy_loss)   

    # Evaluate average distance between predictions and all targets
    avrg_distance = model.avrg_distance(y_pred,X_test)
    print("euclidean average distance: ", avrg_distance)
    print("average distance in cm: ", model.convert_2_cm(avrg_distance, X_test[0])) # Second skeleton in test set

    # Plot prediction
    model.plot(X_test[1], y_pred[0]) # Second skeleton and First prediction

    # Save model parameters & weights
    # model.save_weights("BaselineModel/SGD_model_weights.npy")
    # print("model weights saved")

## MAIN PROGRAM ## (Run only once to initialize model)
if __name__ == "__main__":
    initialize() # Only run this once in the file, weights will be saved in "SGD_model_weights.npy"





