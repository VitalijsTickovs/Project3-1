import numpy as np
import SGD
import time

# TODO : Represent and Visualize current output

# TODO : Make a continuous prediction function
# TODO : Fetch continuous data from camera

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
    
            prediction = model.predict(data)

            # If prediction accuracy is below a certain threshold,
            # for prediction accuracy : I need a way to compare the prediction with the actual output
            # If I don't have a way to evaluate the prediction, I just have to fit the model with the new datastream
            # thus initialization + saving the model would be useless

                # Retrain model with new datastream : 
                # maybe retrain with lower batch-size and lower iterations to improve speed
                # model.fit(data, # data_labels)
                # second_prediction = model.predict(data)

                # Output second_prediction

            # Else, output prediction



