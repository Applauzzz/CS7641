import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import learning_curve

def decision_tree_algorithm(X_train, y_train, X_test, y_test):
    # Model training
    start_time = time.time()
    humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    humidity_classifier.fit(X_train, y_train)
    end_time = time.time()

    # Predictions for test set
    y_predicted = humidity_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_predicted)

    # Predictions for training set
    y_train_predicted = humidity_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_predicted)

    print(f"Decision Tree Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Decision Tree Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Time taken to train: {end_time - start_time:.2f} seconds")

    # Feature Importance:
    plt.figure(figsize=(15, 7))
    plt.barh(X_train.columns, humidity_classifier.feature_importances_)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Decision Tree - Feature Importance')
    plt.show()

    # Learning Curve:
    train_sizes, train_scores, test_scores = learning_curve(humidity_classifier, X_train, y_train, cv=5)
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(15, 7))
    plt.plot(train_sizes, train_scores_mean, label="Training Score", color="b")
    plt.plot(train_sizes, test_scores_mean, label="Cross-Validation Score", color="r")
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Decision Tree - Learning Curve')
    plt.legend(loc="best")
    plt.show()
    print("*" * 100)

    return  test_accuracy
