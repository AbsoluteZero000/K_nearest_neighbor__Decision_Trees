import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

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
        print(total_predictions)
        for i in range(X_test_norm.shape[0]):
            distances = [euclidean_distance(X_test_norm.iloc[i], X_train_norm.iloc[j]) for j in range(X_train_norm.shape[0])]
            nearest_neighbors = np.argsort(distances)[:k]
            y_pred = y_train.iloc[nearest_neighbors]
            prediction = y_pred.value_counts().idxmax()
            if prediction == y_test.iloc[i]:
                correct_predictions += 1

            print(f"Actual: {y_test.iloc[i]}, Predicted (K={k}): {prediction}")

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy (K={k}): {accuracy:.2f}%")
