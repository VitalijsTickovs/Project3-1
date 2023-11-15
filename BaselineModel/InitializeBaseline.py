import numpy as np
from SGD import *
from Skeleton_Dataset.load_skeleton_tracking import *

# TODO : Initialize model parameters and first training 
# TODO : Save model parameters & weights

def initialize_model(lr=0.01, max_iter=1000, batch_size=32, tol=1e-3): 
## Initialize model parameters and first training ##

    # Load & Preprocess data
    dsamp_train, dsamp_test, tr_fea_xyz, tr_label, tr_seq_len, te_fea_xyz, te_label, te_seq_len = preprocess_ucla("BaselineModel/Skeleton_Dataset/ucla_data")
    # print ("no error in loading + preprocessing data")

    # Make sure len(tr_fea_xyz) = len(tr_label) = 1019 = size of training data

    # Create random dataset with 100 rows and 5 columns
    X = np.array(tr_fea_xyz)
    # print("no error in X declaration") 

    # create corresponding target value by adding random noise in the dataset
    # random noise avoids getting stuck in local minima
    y = np.dot(X.T, np.array(tr_label)) + np.random.randn(60, 50, 10) * 0.1
    # print("no error in y declaration")

    # Create an instance of the SGD class
    model = SGD(lr, max_iter, batch_size, tol) # change param if needed
    # print("no error in model declaration")

    model.fit(X, y) # error here : need to adapt SGD.fit to our data

    # Predict using predict method from model
    y_pred = model.predict(X) # Matrix of predicted values
    # print(y_pred) 

    # Save model parameters & weights

## Testing ##
initialize_model(lr=0.01, max_iter=1000, batch_size=32, tol=1e-3)

