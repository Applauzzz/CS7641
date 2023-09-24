import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
import time


def svm_algorithm(X_train, y_train, X_test, y_test):
    # Data Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize variables for plotting
    kernels = ['linear', 'rbf']
    accuracies = []
    training_accuracies = []

    for kernel in kernels:
        start_time = time.time()
        clf = SVC(kernel=kernel, probability=True)
        clf.fit(X_train_scaled, y_train)
        end_time = time.time()

        # Accuracy for test set
        y_pred = clf.predict(X_test_scaled)
        accuracy_svm = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy_svm)
        print(f'SVM Algorithm Test Accuracy with {kernel} kernel: {accuracy_svm * 100:.2f}%')

        # Accuracy for training set
        y_train_pred = clf.predict(X_train_scaled)
        training_accuracy_svm = accuracy_score(y_train, y_train_pred)
        training_accuracies.append(training_accuracy_svm)
        print(f'SVM Algorithm Training Accuracy with {kernel} kernel: {training_accuracy_svm * 100:.2f}%')

        # Time Performance
        print(f'Time taken to train with {kernel} kernel: {end_time - start_time:.2f} seconds')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test_scaled)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'SVM - {kernel.capitalize()} Kernel ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    # Comparison Plot for test accuracy
    plt.figure(figsize=(10, 6))
    x_positions = range(len(kernels))
    plt.bar(x_positions, accuracies, width=0.4, label='Test Accuracy')
    plt.bar([x + 0.2 for x in x_positions], training_accuracies, width=0.4, label='Training Accuracy',
            color='lightblue')
    plt.xticks(x_positions, kernels)
    plt.xlabel('Kernels')
    plt.ylabel('Accuracy')
    plt.title('SVM - Comparison of Kernels')
    plt.legend()
    plt.show()
    print("*" * 100)

    return max(accuracies)


