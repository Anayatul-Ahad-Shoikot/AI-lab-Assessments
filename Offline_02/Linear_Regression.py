import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_diabetes(filename):
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def split_dataset(X, y, train_ratio, val_ratio):
    total_samples = len(X)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_temp, X_val, y_temp, y_val = X[train_size:train_size+val_size], 	X[train_size+val_size:], y[train_size:train_size+val_size], y[train_size+val_size:]
    return X_train, y_train, X_val, y_val, X_temp, y_temp

def train(X_train, y_train, lr, max_iter):
    n_features = X_train.shape[1]
    theta = np.random.rand(n_features + 1)
    history = []
    N_train = len(X_train)
    epsilon = 1e-8
    for itr in range(1, max_iter + 1):
        total_cost = 0
        grad = np.zeros(n_features + 1)
        for i in range(N_train):
            x_i = np.append(X_train[i], 1)
            z = np.dot(x_i, theta)
            h = sigmoid(z)
            J = -y_train[i] * np.log(h + epsilon) - (1 - y_train[i]) * np.log(1 - h + epsilon)
            total_cost += J
            grad += np.dot(x_i, h - y_train[i])
        total_cost /= N_train
        grad /= N_train
        theta -= lr * grad
        history.append(total_cost)
    return history, theta

def predict(X, theta):
    predictions = []
    for i in range(len(X)):
        x_i = np.append(X[i], 1)
        z = np.dot(x_i, theta)
        h = sigmoid(z)
        prediction = 1 if h >= 0.5 else 0
        predictions.append(prediction)
    return predictions



X, y = load_diabetes("diabetes(2).csv")
X_train, y_train, X_val, y_val, X_temp, y_temp = split_dataset(X, y, 0.7, 0.15)
X_test, y_test = X_temp, y_temp
lr = 0.0001
epochs = 200
history, theta = train(X_train, y_train, lr, epochs)
test_predictions = predict(X_test, theta)
test_accurecy = (test_predictions == y_test).mean() * 100
print(f"Test Accuracy with LR: {lr}, Epoch: {epochs}: {test_accurecy:.2f}%")

# ploting grapg
plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()
