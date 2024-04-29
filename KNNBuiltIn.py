import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    data = pd.read_csv("BankNote_Authentication.csv")
    features = data.drop("class", axis=1)
    target = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)


    means = X_train.mean()
    stds = X_train.std()

    X_train_norm = (X_train - means) / stds
    X_test_norm = (X_test - means) / stds
    
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train_norm, y_train)

    y_pred = knn.predict(X_test_norm)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')