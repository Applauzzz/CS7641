import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import learning_curve


def boosting_algorithm(X_train, y_train, X_test, y_test):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # AdaBoost with decision trees
    base_estimator = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=200, random_state=0)

    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    end_time = time.time()

    y_pred = clf.predict(X_test_scaled)

    # Accuracy for the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Boosting Algorithm Test Accuracy: {accuracy * 100:.2f}%")

    # Accuracy for the training set
    y_train_pred = clf.predict(X_train_scaled)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Boosting Algorithm Training Accuracy: {training_accuracy * 100:.2f}%")

    # Time Performance
    print(f"Time taken to train: {end_time - start_time:.2f} seconds")

    # Feature Importance
    plt.figure(figsize=(12, 6))
    plt.barh(X_train.columns, clf.feature_importances_)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Boosting - Feature Importance')
    plt.show()

    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Boosting - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train_scaled, y_train, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Test score')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Boosting - Learning Curve')
    plt.legend(loc='best')
    plt.show()
    print("*" * 100)

    return accuracy
