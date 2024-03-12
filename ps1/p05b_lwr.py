import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    # x_valid = x_valid[ : 20, ]
    # y_valid = y_valid[ : 20]

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    # Get MSE value on the validation set
    y_valid_pred = clf.predict(x_valid)

    print(np.sum((y_valid - y_valid_pred)**2) / y_valid.size)

    #Plot validation predictions on top of training set    
    plt.clf()
    plt.plot(x_valid[:, -1], y_valid, 'bx', label='actual validation')
    plt.plot(x_valid[:, -1], y_valid_pred, 'ro', label='predicted validation')
    plt.legend(loc='upper left')
    
    save_path = "output/img/p05b_lwr_d5.png"

    if save_path is not None:
        plt.savefig(save_path)

    # No need to save predictions
    # Plot data
    
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def weights(self, x):
        m, n = x.shape

        m_og, n_og = self.x.shape

        collec_w = np.zeros((m, m_og , m_og))
        w = np.zeros((m, m_og))

        #for 1st training example
        for j in range(m):
            for i in range(m_og):
                collec_w[j][i][i] = np.exp(-1*((self.x[i] - x[j]).T @ (self.x[i] - x[j])) / (2*(self.tau**2)))
                w[j][i] = collec_w[j][i][i]

        return w

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        m_og, n_og = self.x.shape

        collec_w = np.zeros((m, m_og , m_og))
        w = np.zeros((m, m_og))

        #for 1st training example
        for j in range(m):
            for i in range(m_og):
                collec_w[j][i][i] = np.exp(-1*((self.x[i] - x[j]).T @ (self.x[i] - x[j])) / (2*(self.tau**2)))
                w[j][i] = collec_w[j][i][i]

        self.theta = np.zeros((m, n_og))

        for i in range(m):
            self.theta[i] = np.linalg.inv((self.x.T @ collec_w[i]) @ self.x) @ ((self.y.T @ collec_w[i] )@ self.x)

        y = self.theta * x
        y = np.sum(y, axis=1)

        return y

        # *** END CODE HERE ***
