import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_algorithm(X_train, y_train, X_test, y_test):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # k-NN with different k values
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    accuracies = []
    training_accuracies = []

    for k in k_values:
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        end_time = time.time()

        # Training Accuracy
        y_train_pred = knn.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        training_accuracies.append(train_accuracy)

        # Test Accuracy
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        print(f"k-NN Accuracy with k={k}: {accuracy * 100:.2f}%")
        print(f"Training Accuracy with k={k}: {train_accuracy * 100:.2f}%")
        print(f"Time taken to train with k={k}: {end_time - start_time:.10f} seconds")

    # Plot accuracy  using k values
    plt.figure()
    plt.plot(k_values, accuracies, marker='o', label="Test Accuracy")
    plt.plot(k_values, training_accuracies, marker='x', linestyle='--', label="Training Accuracy")
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.title('k-NN: Accuracy with Different k Values')
    plt.legend()
    plt.show()
    print("*" * 100)

    return max(accuracies)
