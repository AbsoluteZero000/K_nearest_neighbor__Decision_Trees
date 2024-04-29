import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore

def getDistanceFromPoint(point, data):
    return ((point - data) ** 2).sum(axis=1) ** 0.5

if __name__ == "__main__":

    data = pd.read_csv("BankNote_Authentication.csv")
    features = data.drop("class", axis=1)
    target = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)


    means = X_train.mean()
    stds = X_train.std()

    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds

    for k in range(1, 9):
        for i in range(X_test_norm.shape[0]):
            dist = getDistanceFromPoint(X_test_norm.iloc[i], X_train_norm)
            nearest_neighbors = dist.sort_values(ascending=True).head(k).index.tolist()

            y_pred = y_train[nearest_neighbors].values
            counter = 0
            print(y_pred)
            for i in range(y_pred.shape[0]):
                if y_pred[i] == 0:
                    counter -= 1
                else:
                    counter += 1
            y_pred = counter > 0
            
            print(f"Actual: {y_test.iloc[i]}, Predicted (K={k}): {int(y_pred)}")
