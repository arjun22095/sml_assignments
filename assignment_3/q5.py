from typing import Dict, Mapping, Optional, Union, List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt


def find_regression_mse(x_test: npt.NDArray, y_test: npt.NDArray, w: npt.NDArray):
    degree = len(w) - 1
    X = np.column_stack([x_test**i for i in range(degree + 1)])
    y_predicted = X @ w

    y_error = y_test - y_predicted
    y_error_squared = np.square(y_error)
    return np.mean(y_error_squared)


def get_regression_coefficients(
    x_train: npt.NDArray, y_train: npt.NDArray, degree: int
) -> npt.NDArray:
    m: int = len(x_train)
    n: int = degree + 1

    X: npt.NDArray = np.zeros((m, n))  # X is (m, n)
    for i in range(m):
        for j in range(n):
            X[i, j] = x_train[i] ** j

    return np.linalg.inv(X.T @ X) @ X.T @ y_train  # Coefficient vector (degree + 1,)


# Returns average accuracy
def k_fold_cross_validation(
    x_train: npt.NDArray, y_train: npt.NDArray, k: int, degree: int
) -> float:
    print(f"\n{k}-Fold Cross Validation for {degree} Degree Regression Model")

    x_train_split = np.array_split(x_train, k)
    y_train_split = np.array_split(y_train, k)

    errors = []
    for i in range(k):
        x_test = x_train_split[i]  # (20, )
        y_test = y_train_split[i]  # (20, )

        x_train_sub = np.concatenate(
            x_train_split[:i] + x_train_split[i + 1 :]
        )  # (80, )
        y_train_sub = np.concatenate(
            y_train_split[:i] + y_train_split[i + 1 :]
        )  # (80, )
        w = get_regression_coefficients(x_train_sub, y_train_sub, degree)
        errors.append(find_regression_mse(x_test, y_test, w))
        print(f"{i + 1}th Fold as Test Data : MSE = {errors[-1]}")

    print(f"Average {k}-Fold Cross Validation MSE = {np.mean(errors)}\n")
    return np.mean(errors)


def find_best_degree(
    x_train: npt.NDArray, y_train: npt.NDArray, k: int, max_degree: int
) -> int:
    degree_errors = []
    for degree in range(1, max_degree + 1):
        avg_k_fold_cross_validation_error = k_fold_cross_validation(
            x_train, y_train, k, degree
        )
        degree_errors.append(avg_k_fold_cross_validation_error)

    return np.argmin(degree_errors) + 1


def get_training_data(N: int, l: float, r: float) -> Tuple[npt.NDArray, npt.NDArray]:
    rng = np.random.default_rng()
    x_train: npt.NDArray = rng.uniform(l, r, N)
    y_train: npt.NDArray = np.sin(x_train)
    gaussian_noise: npt.NDArray = rng.normal(loc=0, scale=0.1, size=N)
    y_train = y_train + gaussian_noise
    return (x_train, y_train)


def print_polynomial(w: npt.NDArray):
    for i in range(len(w)):
        if i == 0:
            print(f"{w[i]} ", end="")
        else:
            print(f"+ ({w[i]})x^{i} ", end="")
    print()


def main():
    l, r = 0, 2 * np.pi
    x_train, y_train = get_training_data(N=100, l=l, r=r)
    degree: int = find_best_degree(x_train, y_train, k=5, max_degree=4)
    print(
        "The best degree for our regression model = ",
        degree,
        "as it had the lowest average MSE when we ran k-fold cross validation",
    )

    w: npt.NDArray = get_regression_coefficients(x_train, y_train, degree)
    print("The regression polynomial is :")
    print_polynomial(w)

    # Plot everything
    plt.figure(figsize=(10, 6))
    x_plot = np.linspace(l, r, 50)

    # Plot the regression prediction
    X_plot = np.column_stack([x_plot**i for i in range(degree + 1)])
    y_plot = X_plot @ w
    plt.plot(x_plot, y_plot, color="blue", label="Regression model prediction")

    # Plot the actual function, sinx
    y_plot = [np.sin(x) for x in x_plot]
    plt.plot(x_plot, y_plot, color="green", label="sin(x) Actual")

    # Plot the noisy training data
    plt.scatter(x_train, y_train, color="red", alpha=0.5, label="Noisy training data")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Regression Fit to sin(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
