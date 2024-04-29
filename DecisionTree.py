import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore


def experiment(trainPercent, n):
    accuracies = []
    treeSizes = []
    for _ in range(n):
        XTrain, XTest, yTrain, yTest = train_test_split(features, target, test_size=1 - trainPercent)

        model = DecisionTreeClassifier()
        model.fit(XTrain, yTrain)

        y_pred = model.predict(XTest)
        accuracy = accuracy_score(yTest, y_pred)
        treeSize = model.tree_.node_count
        accuracies.append(accuracy)
        treeSizes.append(treeSize)


    meanAccuracy = sum(accuracies) / len(accuracies)
    minAccuracy = min(accuracies)
    maxAccuracy = max(accuracies)

    meanTreeSize = sum(treeSizes) / len(treeSizes)
    minTreeSize = min(treeSizes)
    maxTreeSize = max(treeSizes)

    return meanAccuracy, minAccuracy, maxAccuracy, meanTreeSize, minTreeSize, maxTreeSize


if __name__ == "__main__":

    trainPercent = 0.25
    n = 5

    data = pd.read_csv("BankNote_Authentication.csv")
    features = data.drop("class", axis=1)
    target = data["class"]


    for _ in range(n):
        meanAccuracy, minAccuracy, maxAccuracy, meanTreeSize, minTreeSize, maxTreeSize = experiment(trainPercent, n)

        print(f"Train size: {trainPercent*100:.0f}%")
        print(f"Accuracy: Mean - {meanAccuracy:.4f}, Min - {minAccuracy:.4f}, Max - {maxAccuracy:.4f}")
        print(f"Tree size: Mean - {meanTreeSize:.0f}, Min - {minTreeSize:.0f}, Max - {maxTreeSize:.0f}")
        print("-"*50)

    split_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    n = 5

    overallMeanTreesizes = []
    overallMeanAccuracies = []
    for split in split_ratios:
        meanAccuracies = []
        meanTreeSizes = []

        overallMinAccuracy = 100000000
        overallMinTreeSize = 100000000
        overallMaxAccuracy = 0
        overallMaxTreeSize = 0

        for _ in range(n):
            randomSeed = 3
            meanAccuracy, minAccuracy, maxAccuracy, meanTreeSize, minTreeSize, maxTreeSize = experiment(split, n)
            overallMinAccuracy = min(overallMinAccuracy, minAccuracy)
            overallMaxAccuracy = max(overallMaxAccuracy, maxAccuracy)
            overallMaxTreeSize = max(overallMaxTreeSize, maxTreeSize)
            overallMinTreeSize = min(overallMinTreeSize, minTreeSize)
            meanAccuracies.append(meanAccuracy)
            meanTreeSizes.append(meanTreeSize)


        overallMeanAccuracy = sum(meanAccuracies) / len(meanAccuracies)
        overallMeanTreeSize = sum(meanTreeSizes) / len(meanTreeSizes)

        overallMeanAccuracies.append(overallMeanAccuracy)
        overallMeanTreesizes.append(overallMeanTreeSize)

        print(f"Train size: {split*100:.0f}%")
        print(f"Accuracy: Mean - {overallMeanAccuracy:.4f}, Min - {overallMinAccuracy:.4f}, Max - {overallMaxAccuracy:.4f}")
        print(f"Tree size: Mean - {overallMeanTreeSize:.0f}, Min - {overallMinTreeSize:.0f}, Max - {overallMaxTreeSize:.0f}")
        print("-"*50)

    plt.scatter(split_ratios, overallMeanTreesizes, label="Accuracy")
    plt.xlabel("Train size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.scatter(split_ratios, overallMeanAccuracies, label="Tree size")
    plt.xlabel("Train size")
    plt.ylabel("Tree size")
    plt.legend()
    plt.show()
