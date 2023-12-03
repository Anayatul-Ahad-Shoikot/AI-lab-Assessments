import numpy as np
import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            images.append(img.flatten())  # Flatten the image into a 1D array
    return np.array(images)


train_folder = '/kaggle/input/vegetable-image-dataset/Vegetable Images/train'
test_folder = '/kaggle/input/vegetable-image-dataset/Vegetable Images/test'
X_train = []
Y_train = []
X_test = []
Y_test = []
for class_index, class_folder in enumerate(os.listdir(train_folder)):
    class_path = os.path.join(train_folder, class_folder)
    images = load_images_from_folder(class_path)
    X_train.extend(images)
    Y_train.extend([class_index] * len(images))
for class_index, class_folder in enumerate(os.listdir(test_folder)):
    class_path = os.path.join(test_folder, class_folder)
    images = load_images_from_folder(class_path)
    X_test.extend(images)
    Y_test.extend([class_index] * len(images))
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


class KNN:
    def __init__(self, k):
        self.k = k
    def train(self, X_train, Y_train, X_test):
        num_test = X_test.shape[0]
        num_train = X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(X_train - X_test[i, :]), axis=1))
        Y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            sorted_indices = np.argsort(dists[i, :])
            closest_y = Y_train[sorted_indices[:self.k]]
            Y_pred[i] = np.argmax(np.bincount(closest_y))
        return Y_pred.astype(int)



knn_classifier = KNN(k=3)
predicted_labels = knn_classifier.train(X_train, Y_train, X_test)
accuracy = np.mean(predicted_labels == Y_test) * 100
print(f"Accuracy of KNN classifier: {accuracy:.5f}%")
