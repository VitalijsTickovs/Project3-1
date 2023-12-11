import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def normalize_data(df):

    df = np.array(df)

    keypoint_means = np.mean(df)
    keypoint_stddevs = np.std(df)

    normalized_df = (df - keypoint_means) / keypoint_stddevs

    return normalized_df.tolist()

# MAIN:
if __name__ == "__main__":
    pass