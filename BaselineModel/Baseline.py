import numpy as np
import SGD
import time
import random

# TODO : Make a continuous prediction function
# TODO : Fetch continuous data from camera

THRESHOLD = 0.5

def fetch_data():
    # Fetch continuous data from camera
    # Apply preprocessing to data
    # specifically fetch skeleton keypoints list of 34 keypoints
    pass

## MAIN PROGRAM ##
if __name__ == "__main__":

    # Load the initialized model
    model = SGD.SGD()
    model.load_weights("BaselineModel/SGD_model_weights.npy")

    previous_time = time.time()
    timestep = 5 # seconds

    while True:
        current_time = time.time()
        elapsed_time = current_time - previous_time

        if elapsed_time > timestep:
            data = fetch_data()
    
            # Compute and evaluate prediction
            prediction = model.predict(data)
            l1_loss,cross_entropy_loss = model.evaluate_loss(prediction, data)

            # If prediction accuracy is below a certain threshold,
            if l1_loss < THRESHOLD or cross_entropy_loss < THRESHOLD:
                # Output prediction
                print(prediction)
            
            # retrain model with new datastream
            else: 
                model.fit(data, data)
                second_prediction = model.predict(data)
                # output prediction

            

          

             

        



