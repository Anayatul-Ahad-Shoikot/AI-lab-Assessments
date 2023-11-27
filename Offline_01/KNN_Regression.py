import numpy as np
import random
import math

data = np.genfromtxt('diabetes.csv', delimiter=',')
data.tolist()

train_data = []
validation_data = []
test_data = []

def data_preparation(train_size,val_size,test_size):
  random.shuffle(data)
  for s in data:
    r = random.random()
    if r>=0 and r<=train_size:
      train_data.append(s)
    elif r>train_size and r<= train_size+val_size:
      validation_data.append(s)
    else:
      test_data.append(s)

data_preparation(.7, .15, .15)

def euclidean_dist(v,t):
  d = 0
  for x,y in zip(v,t):
    d += (x-y)**2
  return math.sqrt(d)

def sort_data(d):
  return d[1]

def KNN_Regressor(k, train, val):
  Error = 0
  distance = []
  for v in val:
    for t in train:
      dist = euclidean_dist(v[:-1], t[:-1])
      distance.append([t, dist])
    distance.sort(key = sort_data)
    select = distance[:k]
    Sum = 0
    for s in select:
      Sum = Sum + s[0][1]
    Sum = Sum/k
    Error = Error + (v[-1] - Sum)**2
  return Error/len(val)

print ("Training Dataset Result: ", KNN_Regressor(1, train_data, validation_data))
print("Testing Dataset Result: ", KNN_Regressor(1, train_data, test_data))
