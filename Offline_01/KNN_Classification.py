import numpy as np
import random
import math

data = np.genfromtxt('iris.csv', delimiter=',')
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

def kNN_classifier(k, train, val):
  distance = []
  accuracy = 0
  for v in val:
    for t in train:
      dist = euclidean_dist(v[:-1],t[:-1])
      distance.append([t,dist])
    distance.sort(key=sort_data)
    select = distance[:k]
    label = {
        0:0,
        1:0,
        2:0
    }
    for s in select:
      label[s[0][-1]] += 1
    predicted_class = max(label,key=label.get)
    if v[-1]-predicted_class == 0:
      accuracy += 1
  return accuracy/len(val)


print("Training dataset result: ", kNN_classifier(5, train_data, validation_data))
print("Testing dataset result: ", kNN_classifier(5, train_data, test_data))
