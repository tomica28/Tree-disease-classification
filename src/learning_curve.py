import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
def draw_learing_curve(estimator, X_train, y_train):
    train_sizes, train_scores, test_scores =\
        learning_curve(estimator=estimator, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 50), cv=10, n_jobs=2)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean,color='blue', marker='o',markersize=5, label='Dokladnosc uczenia')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='Dokladnosc walidacji')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Liczba probek uczacych')
    plt.ylabel('Dokladnosc')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.1])
    plt.show()