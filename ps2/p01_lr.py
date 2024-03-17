# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt

norm_theta = []
thetas = []
grads = []
it =  []
total_it = 10000


def plot_grad(it, grads, color):
    # plt.clf()
    plt.figure(1)
    plt.plot(it, grads[:,0], color, linewidth=2)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('grad')
    
    return plt

def plot_theta(it, thetas, color):
    # plt.clf()
    plt.figure(2)
    plt.plot(it, thetas[:len(it),0], color, linewidth=2)
    # plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('theta')
    
    return plt

def plot_norm_theta(it, norm_thetas, color):
    # plt.clf()
    plt.figure(3)
    plt.plot(it, norm_thetas, color, linewidth=2)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('norm theta')
    
    return plt

def plot_classification_data(x, y):
    plt.clf()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y, total_it):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)

            # #for plotting gradients
            # plt1 = plot_grad(it, grads, 'go')
            # plt2 = plot_theta(it, thetas, 'go')
            # plt3 = plot_norm_theta(it, norm_theta, 'go')

            break

        #for plotting gradients and thetas
        # if i < total_it:
        #     it[i] = i
        #     grads[i] = grad
        #     thetas[i] = theta
        #     norm_theta[i] = np.linalg.norm(prev_theta - theta)
        # if i == total_it:
        #     plt1 = plot_grad(it, grads, 'rx')
        #     plt2 = plot_theta(it, thetas, 'rx')
        #     plt3 = plot_norm_theta(it, norm_theta, 'rx')
        #     plt1.show()
        #     plt2.show()
        #     plt3.show()
    return


def main():
    global grads, it, thetas, norm_theta, total_it
    total_it = 50000


    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('data/ds1_a.csv', add_intercept=True)

    # grads = np.zeros((total_it, Xa.shape[1]))
    # thetas = np.zeros((total_it, Xa.shape[1]))
    # it = np.zeros(total_it)
    # norm_theta = np.zeros(total_it)

    logistic_regression(Xa, Ya)



    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('data/ds1_b.csv', add_intercept=True)

    # grads = np.zeros((total_it, Xa.shape[1]))
    # thetas = np.zeros((total_it, Xa.shape[1]))
    # it = np.zeros(total_it)
    # norm_theta = np.zeros(total_it)

    logistic_regression(Xb, Yb)
    # plot_classification_data(Xb, Yb)


if __name__ == '__main__':
    main()[:len(it),0]