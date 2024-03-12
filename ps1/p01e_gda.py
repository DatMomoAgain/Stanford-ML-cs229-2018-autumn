import numpy as np
import util
import matplotlib.pyplot as plt


from linear_model import LinearModel
from p01b_logreg import main as LogReg

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)


    # *** START CODE HERE ***
    theta1 = LogReg(train_path, eval_path, pred_path)
    clf = GDA()

    theta2 = clf.fit(x_train, y_train)

    y_predict = clf.predict(x_valid)

    img_save_path = ""
    if (eval_path == "data/ds2_valid.csv"):
        img_save_path = 'output/img/p01e_gda_d2.png'
    elif (eval_path == "data/ds1_valid.csv"):
        img_save_path = 'output/img/p01e_gda_d1.png'

    plt.clf()

    util.plot2(x_valid, y_valid, theta1, theta2, img_save_path)

    np.savetxt(pred_path, y_predict, fmt='%d')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y

        m, n = x.shape

        phi = 0
        for i in range(m):
            if y[i]:
                phi += 1
        phi = 1/m * phi

        mu0 = np.zeros(n)
        d0 = 0
        mu1 = np.zeros(n)
        d1 = 0
        
        for i in range(m):
            if y[i]:
                mu1 += x[i]
                d1 += 1
            else:
                mu0 += x[i]
                d0 += 1

        mu0 = mu0 / d0
        mu1 = mu1 / d1

        cov = np.zeros((n, n))

        for i in range(m):
            if y[i]:
                cov += np.outer(x[i] - mu1, x[i] - mu1)
            else: 
                cov += np.outer(x[i] - mu0, x[i] - mu0)

        cov = 1/m * cov

        covInv = np.linalg.inv(cov)

        self.theta = np.zeros(n+1)

        #beware
        self.theta[0] = 0.5*((mu0.T @ covInv @ mu0) - (mu1.T @ covInv @ mu1) + np.log((1 - phi) / (phi)))
        self.theta[1 : n+1] = covInv @ (mu1 - mu0)

        return self.theta
        
        # *** END CODE HERE ***
    
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        py = 1 / (1 + np.exp(-1 * ((x @ self.theta[1 : n+1]) + self.theta[0])))

        ans = np.zeros(m)

        ans[py > 0.5] = 1

        return ans

        # *** END CODE HERE
