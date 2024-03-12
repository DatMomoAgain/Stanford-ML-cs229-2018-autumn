import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***    
    # Search tau_values for the best tau (lowest MSE on the validation set)

    mse = 1000
    best_tau = 0

    for i in range(len(tau_values)):
        clf = LocallyWeightedLinearRegression(tau_values[i])
        clf.fit(x_train, y_train)
        y_valid_pred = clf.predict(x_valid)
        
        tmse = np.sum((y_valid - y_valid_pred)**2) / y_valid.size
        if(tmse < mse):
            mse = tmse
            best_tau = tau_values[i]

    # Fit a LWR model with the best tau value
    clf = LocallyWeightedLinearRegression(best_tau)
    clf.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    y_test_pred = clf.predict(x_test)
    mse = np.sum((y_test - y_test_pred)**2) / y_test.size
    print("lowest mse: ", mse)
    print("best tau: ", best_tau)

    # Save predictions to pred_path
    np.savetxt(pred_path, y_test_pred, fmt='%f')

    plt.clf()

    # Plot data
    plt.plot(x_test[:,-1], y_test, 'bx', label='actual test')
    plt.plot(x_test[:, -1], y_test_pred, 'ro', label='predicted test')
    plt.legend(loc='upper left')
    
    save_path = "output/img/p05b_tau_d5.png"

    if save_path is not None:
        plt.savefig(save_path)

    # *** END CODE HERE ***
