import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore

def mean(df):
    # Calculate the mean for each column in the DataFrame
    mean_values = {}
    for column in df.columns:
        column_mean = sum(df[column]) / len(df[column])
        mean_values[column] = column_mean
    return mean_values

def standard_deviation(df):
    # Calculate the standard deviation for each column in the DataFrame
    std_values = {}
    for column in df.columns:
        column_mean = sum(df[column]) / len(df[column])
        squared_diff = [(x - column_mean) ** 2 for x in df[column]]
        column_std = (sum(squared_diff) / len(df[column])) ** 0.5
        std_values[column] = column_std
    return std_values

def getDistanceFromPoint(point, data):
    return ((point - data) ** 2).sum(axis=1) ** 0.5

def custom_shuffle(indices):
    n = len(indices)
    permutation = np.random.permutation(n)
    for i in range(n):
        j = permutation[i]
        indices[i], indices[j] = indices[j], indices[i]
    return indices

def t_t_s(X, y, test_size=0.3, random_state=None):
    if len(X) != len(y):
        raise ValueError("features and target must have the same length")
    
    test_samples = int(len(X) * test_size)
    
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    if random_state is not None:
        np.random.seed(random_state)
        
    indices = np.arange(len(X))
    shuffled_indices = custom_shuffle(indices)

    train_indices = shuffled_indices[test_samples:]
    test_indices = shuffled_indices[:test_samples]

    X_train, X_test = X_reset.iloc[train_indices], X_reset.iloc[test_indices]
    y_train, y_test = y_reset.iloc[train_indices], y_reset.iloc[test_indices]

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":

    data = pd.read_csv("BankNote_Authentication.csv")
    features = data.drop("class", axis=1)
    target = data["class"]
    X_train, X_test, y_train, y_test = t_t_s(features, target, test_size=0.3, random_state=42)
    means = X_train.mean()
    means = mean(X_train)
    stds = standard_deviation(X_train)

    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds

    for k in range(1, 9):
        correct_predictions = 0
        total_predictions = len(y_test)
        for i in range(X_test_norm.shape[0]):
            dist = getDistanceFromPoint(X_test_norm.iloc[i], X_train_norm)
            print("Distances: ")
            print(dist)
            nearest_neighbors = dist.sort_values(ascending=True).head(k).index.tolist()
            print("nearest neighbours: ")
            print(nearest_neighbors)
            y_pred = y_train[nearest_neighbors].values
            counter = 0
            for i in range(y_pred.shape[0]):
                if y_pred[i] == 0:
                    counter -= 1
                else:
                    counter += 1
            y_pred = counter > 0

            if int(y_pred) == y_test.iloc[i]:
                correct_predictions += 1

            print(f"Actual: {y_test.iloc[i]}, Predicted (K={k}): {int(y_pred)}")

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy (K={k}): {accuracy:.2f}%")