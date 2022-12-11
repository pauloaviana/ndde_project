import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression():
    # define the data
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    # calculate the mean of the x and y values
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # calculate the variance and covariance
    var_x = np.var(x)
    cov_xy = np.cov(x, y)[0][1]

    # calculate the coefficients
    b1 = cov_xy / var_x
    b0 = mean_y - (b1 * mean_x)

    # print the coefficients
    print("b0 =", b0)
    print("b1 =", b1)

if __name__ == '__main__':
    linear_regression()
