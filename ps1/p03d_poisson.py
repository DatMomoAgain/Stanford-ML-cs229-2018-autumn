import numpy as np
import util
import matplotlib.pyplot as plt


from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    y = clf.predict(x_valid)
    np.savetxt(pred_path, y, fmt='%f')

    plt.clf()

    plt.plot(y_valid, 'go', label='label')
    plt.plot(y, 'rx', label='prediction')
    plt.legend(loc='upper left')

    save_path = "output/img/p03d_poisson_d4.png"

    if save_path is not None:
        plt.savefig(save_path)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y

        m, n = x.shape
        self.theta = np.zeros(n)

        h = np.exp(x @ self.theta)

        for i in range(m):
            h = np.exp(self.theta.T @ x[i])
            self.theta = self.theta + (self.step_size / m) *(self.y[i] - h)*x[i]
            # print(self.theta)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        h = np.exp(x @ self.theta)

        return h

        # *** END CODE HERE ***
