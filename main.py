from model.decision_tree_module import  decision_tree_algorithm
from model.nn_module import nn_algorithm
from model.boosting_module import boosting_algorithm
from model.svm_module import svm_algorithm
from model.knn_module import knn_algorithm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def main():

    # Loading and cleaning data
    data = pd.read_csv('./daily_weather.csv')
    del data['number']
    data = data.dropna()

    # Creating target column for high humidity
    data['high_humidity_label'] = (data['relative_humidity_3pm'] > 24.99) * 1

    # Features for prediction
    morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
                        'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
                        'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']

    X = data[morning_features]
    y = data['high_humidity_label']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

    # Call decision tree algorithm
    decision_tree_accuracy = decision_tree_algorithm(X_train, y_train, X_test, y_test)

    # Call neural network algorithm
    neural_network_accuracy = nn_algorithm(X_train, y_train, X_test, y_test)

    # Call boosting algorithm
    boosting_accuracy = boosting_algorithm(X_train, y_train, X_test, y_test)

    # Call SVM algorithm
    svm_accuracy = svm_algorithm(X_train, y_train, X_test, y_test)

    # Call k-NN algorithm
    knn_accuracy = knn_algorithm(X_train, y_train, X_test, y_test)


    # print(neural_network_accuracy)

    algorithms = ['Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'k-NN']
    accuracies = [decision_tree_accuracy, neural_network_accuracy, boosting_accuracy, svm_accuracy, knn_accuracy]

    colors = ['blue', 'green', 'red', 'purple', 'orange']


    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, accuracies, color=colors, width=0.6)
    plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])

    # Label with the exact accuracy above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom',
                 color='black')

    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Different Algorithms')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
