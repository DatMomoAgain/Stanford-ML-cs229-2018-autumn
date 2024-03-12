import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)

    y_predict = clf.predict(x_valid)

    # img_save_path = ""
    # if (eval_path == "data/ds2_valid.csv"):
    #     img_save_path = 'output/img/p01b_logreg_d2.png'
    # elif (eval_path == "data/ds1_valid.csv"):
    #     img_save_path = 'output/img/p01b_logreg_d1.png'

    # util.plot(x_valid, y_valid, theta, img_save_path)
    
    np.savetxt(pred_path, y_predict, fmt='%d')

    return theta

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.x = x
        self.y = y
        self.theta = np.zeros(n)
        thetaPrev = np.ones(n)

        ldash = np.zeros(n)
        hess = np.zeros((n, n))

        c = 0

        while(abs(self.theta - thetaPrev).all() > self.eps):

            ldash = np.zeros(n)
            hess = np.zeros((n, n))

            thetaPrev = self.theta

            for i in range(m):
                k = self.theta.T @ self.x[i]
                # if(np.isnan(k) == False and c == 1):
                #     print("k: ", k)

                htheta = 1 / (1 + np.exp(-1*k))
                ldash += -1 * (y[i] - htheta) * x[i]
                hess += htheta * (1 - htheta) * np.outer(self.x[i], self.x[i])

            hess = 1/m * hess
            ldash = 1/m * ldash
            self.theta = self.theta - np.linalg.inv(hess) @ ldash
            c += 1
            
            # print("theta: ", self.theta)
            # print("ldash: ", ldash)
            # print("hess: ", hess)
            # print("hess inverse: ", np.linalg.inv(hess))
            # print("\n\n")
        
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
        htheta = 1 / (1 + np.exp(-1 * x @ self.theta))
        
        ans = np.zeros(m)
        ans[htheta > 0.5] = 1

        return ans

        # *** END CODE HERE ***
