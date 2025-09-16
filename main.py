import pandas
import numpy as np
import matplotlib.pyplot as plt

'''
train = pandas.read_csv("data/train.csv")
train = train.drop(['id','RhythmScore','AudioLoudness','VocalContent','AcousticQuality','InstrumentalScore','LivePerformanceLikelihood',"TrackDurationMs", "MoodScore"], axis=1)
train = train.iloc[0:2000]
train.to_csv("data/cleaned_train.csv",index=False)

train = pandas.read_csv("data/test.csv")
train = train.drop(['id','RhythmScore','AudioLoudness','VocalContent','AcousticQuality','InstrumentalScore','LivePerformanceLikelihood',"TrackDurationMs", "MoodScore"], axis=1)
train = train.iloc[0:2000]
train.to_csv("data/cleaned_test.csv",index=False)
'''

train = pandas.read_csv("cleaned_train.csv")
test = pandas.read_csv("cleaned_test.csv")


X = train.drop('BeatsPerMinute',axis=1).to_numpy().reshape(-1,1)
y = train["BeatsPerMinute"].to_numpy()


theta_0 = 0.0
theta_1 = 0.0

learning_rate = 0.01

iterations = 100

costs = []

m = len(X)

print(f'''Starting values:
theta_0: {theta_0}
theta_1: {theta_1}
learning_rate: {learning_rate}
iterations: {iterations}
cost: 0
''')

for i in range(iterations):
    predictions = theta_0 + theta_1 * X
    errors = predictions - y

    cost = (1/(2*m)) * np.sum(errors**2)
    costs.append(cost) # Store the cost

    gradient_theta_0 = (1/m) * np.sum(errors)
    gradient_theta_1 = (1/m) * np.sum(errors * X)

    theta_0 = theta_0 - learning_rate * gradient_theta_0
    theta_1 = theta_1 - learning_rate * gradient_theta_1



print(f"theta_0: {theta_0}")
print(f"theta_1: {theta_1}")

print("Cost:",cost)
