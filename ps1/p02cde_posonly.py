import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col='t', add_intercept=False)

    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    _, t_test = util.load_dataset(test_path, label_col='t', add_intercept=False)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    _, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=False)


    # Part (c): Train and test on true labels
    clf = LogisticRegression()
    theta = clf.fit(x_train, t_train)
    y_test_predict = clf.predict(x_test)

    # Make sure to save outputs to pred_path_c
    np.savetxt(pred_path_c, y_test_predict, fmt='%d')
    util.plot(x_test, t_test, theta, "output/img/p02c_posonly_d3")



    # Part (d): Train on y-labels and test on true labels
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)
    y_test_predict = clf.predict(x_test)

    # Make sure to save outputs to pred_path_d
    np.savetxt(pred_path_d, y_test_predict, fmt='%d')
    util.plot(x_test, t_test, theta, "output/img/p02d_posonly_d3")



    # Part (e): Apply correction factor using validation set and test on true labels
    htheta_valid = 1 / (1 + np.exp(-1 * x_valid @ theta))
    sum_h = np.sum(htheta_valid[y_valid == 1])
    count_v = sum(y_valid == 1)
    alpha = sum_h / count_v

    htheta_test = 1 / (1 + np.exp(-1 * x_test @ theta))
    htheta_test = htheta_test / alpha
    y_test_predict[htheta_test >= 0.5] = 1    
    
    # Plot and use np.savetxt to save outputs to pred_path_e
    np.savetxt(pred_path_e, y_test_predict, fmt='%d')
    util.plot(x_test, t_test, theta, "output/img/p02e_posonly_d3", alpha)

    # *** END CODER HERE
