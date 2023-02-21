import math

import pandas as pd

df = pd.read_csv('dataset/flags/maskDetectorData.csv', header=None)

count = len(df)
no_steps = math.ceil(count/16)

print("Count of annotations: {}".format(count))
print("Number of steps per epoch: {}".format(no_steps))