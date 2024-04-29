import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

if __name__ == "__main__":

    data = pd.read_csv("BankNote_Authentication.csv")
    features = data.drop("class", axis=1)
    target = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)


    means = X_train.mean()
    stds = X_train.std()

    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds

    def KNN(X_train, y_train, X_test, k=3):
        predictions = []
        for test_point in X_test.values:
            distances = []
            for train_point in X_train.values:
                dist = euclidean_distance(test_point, train_point)
                distances.append(dist)
            
            nearest_indices = np.argsort(distances)[:k]
            
            nearest_labels = [y_train.iloc[i] for i in nearest_indices]
            
            label_counts = Counter(nearest_labels)
            prediction = max(label_counts, key=label_counts.get)
            predictions.append(prediction)
        
        return predictions

    k = 5
    y_pred = KNN(X_train_norm, y_train, X_test_norm, k)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")