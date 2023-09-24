import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

def nn_algorithm(X_train, y_train, X_test, y_test):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Data Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize variables for plotting
    iterations = list(range(200, 2001, 100))
    accuracies = []
    training_accuracies = []
    times = []

    for max_iter in iterations:
        start_time = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=max_iter, random_state=0)
        mlp.fit(X_train_scaled, y_train)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # Training Accuracy
        y_train_pred = mlp.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        training_accuracies.append(train_accuracy)

        # Test Accuracy
        y_pred = mlp.predict(X_test_scaled)
        accuracy_nn = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy_nn)

        print(f'NN Algorithm Accuracy with {max_iter} iterations: {accuracy_nn * 100:.2f}%')
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Time taken to train: {end_time - start_time:.2f} seconds")

    # Plot Time vs Iterations
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(iterations, times, 'g')
    plt.xlabel('Iterations')
    plt.ylabel('Time (seconds)')
    plt.title('NN - Effect of Increasing Iterations on Time Performance')
    plt.grid(True)

    # Plot Accuracy vs Iterations
    plt.subplot(3, 1, 2)
    plt.plot(iterations, accuracies, 'b', label="Test Accuracy")
    plt.plot(iterations, training_accuracies, 'r--', label="Training Accuracy")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.title('NN - Effect of Increasing Iterations on Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, mlp.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('NN - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    print("*" * 100)

    return max(accuracies)
