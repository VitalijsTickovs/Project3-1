import numpy as np
import SGD
import time

# TODO : Represent and Visualize current output
# TODO : Adapt SGD model to fit our data and not UCLA data


# TODO : Make a continuous prediction function
# TODO : Fetch continuous data from camera

def fetch_data():
    # Fetch continuous data from camera
    # Apply preprocessing to data
    pass

## MAIN PROGRAM ##
if __name__ == "__main__":

    # Load the initialized model
    model = SGD.SGD()
    model.load_weights("Project3-1/BaselineModel/SGD_model_weights.npy")

    previous_time = time.time()
    timestep = 1 # seconds

    while True:
        current_time = time.time()
        elapsed_time = current_time - previous_time

        if elapsed_time > timestep:
            data = fetch_data()
            prediction = model.predict(data)

            # If prediction accuracy is below a certain threshold,
            
                # Retrain model with new datastream : 
                # maybe retrain with lower batch-size and lower iterations to improve speed
                # model.fit(data, # data_labels)
                # second_prediction = model.predict(data)

                # Output second_prediction

            # Else, output prediction



