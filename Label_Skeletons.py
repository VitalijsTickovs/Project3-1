import pickle
import pandas as pd
import numpy as np

# loading model
with open('models/clf34.pkl', 'rb') as f:
    clf = pickle.load(f)

# loading scaler
with open('models/scaler34.pkl', 'rb') as f:
    scaler = pickle.load(f)


def center_skeleton(keypoints):
    # Assuming the row is a pandas Series with keypoint columns like 'keypoint1_x', 'keypoint1_y', 'keypoint1_z', etc.
    centered = keypoints.copy()
    ref_x, ref_y, ref_z = keypoints['keypoint0_x'], keypoints['keypoint0_y'], keypoints['keypoint0_z']

    # Iterate through each keypoint and center them
    for i in range(0, 34):  # Adjust the range based on the number of keypoints
        centered[f'keypoint{i}_x'] -= ref_x
        centered[f'keypoint{i}_y'] -= ref_y
        centered[f'keypoint{i}_z'] -= ref_z

    return centered


def label_skeleton(keypoints):
    centered = center_skeleton(keypoints)
    centered = np.array(centered).reshape(1,-1)
    scaled = scaler.transform(centered)

    pred = clf.predict(scaled)
    return pred[0]