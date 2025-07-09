import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table


def compute_loss(y_predicted, y_actual, loss_type: str) -> npt.NDArray:
    y_actual = np.asarray(y_actual)

    if not isinstance(y_predicted, np.ndarray):
        y_predicted = np.full_like(y_actual, y_predicted)

    if loss_type == "absolute":
        return np.abs(y_actual - y_predicted)
    elif loss_type == "squared":
        return ((y_actual - y_predicted) ** 2) / 2
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def generate_stump(x_train, y_train, loss_type: str):
    best_loss = float("inf")
    best_split_mid = 0
    best_left_predict = 0
    best_right_predict = 0

    unique_vals = sorted(np.unique(x_train))
    for i in range(len(unique_vals) - 1):
        mid = (unique_vals[i] + unique_vals[i + 1]) / 2

        left_mask = x_train < mid
        left_labels = y_train[left_mask]
        left_predict = np.mean(left_labels)
        left_loss = compute_loss(y_predicted=left_predict, y_actual=left_labels, loss_type=loss_type)

        right_labels = y_train[~left_mask]
        right_predict = np.mean(right_labels)
        right_loss = compute_loss(y_predicted=right_predict, y_actual=right_labels, loss_type=loss_type)

        # total_loss = (left_loss * len(left_data) + right_loss * len(right_data)) / len(dataset)
        # No need to weight both of them
        total_loss = np.sum(left_loss) + np.sum(right_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            best_split_mid = mid
            best_left_predict = left_predict
            best_right_predict = right_predict

    if best_loss == float("inf"):
        # print("Returning none")
        return None

    def best_split_condition(x: float) -> float:
        if x < best_split_mid:
            return best_left_predict
        else:
            return best_right_predict

    return best_split_condition


class GradientBoost:
    def __init__(self, n_iterations: int, loss_type: str, learning_rate: float):
        self.n_iterations: int = n_iterations
        self.loss_type: str = loss_type
        self.learning_rate: float = learning_rate
        self.classifiers = []
        self.learning_rates = []

    def predict(self, x: np.ndarray, depth: int = None) -> npt.NDArray:
        if (depth is None) or (depth < 0) or (depth > len(self.classifiers)):
            depth = len(self.classifiers)

        prediction = np.zeros_like(x, dtype=float)
        for i in range(depth):
            prediction += self.learning_rates[i] * np.array([self.classifiers[i](xj) for xj in x])

        return prediction

    def compute_negative_gradient_of_loss(self, y_predicted, y_actual, loss_type: str) -> npt.NDArray:
        # Terminology: y_predicted = ỹ, and y_actual = y
        y_actual = np.asarray(y_actual)

        if not isinstance(y_predicted, np.ndarray):
            y_predicted = np.full_like(y_actual, y_predicted)

        if loss_type == "absolute":
            return -np.sign(y_predicted - y_actual)
        elif loss_type == "squared":
            return -1 * (y_predicted - y_actual)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        # Initial prediction = mean
        self.classifiers.clear()
        self.classifiers.append(lambda a: np.mean(y_train))

        # If inititally I wanted decision stump
        """
        self.classifiers.append(
            generate_stump(
                x_train=x_train,
                y_train=y_train,
                loss_type=self.loss_type,
            )
        )
        """

        self.learning_rates.append(1)
        for i in range(self.n_iterations - 1):
            predicted: npt.NDArray = self.predict(x_train)
            negative_grad_loss_vector: npt.NDArray = self.compute_negative_gradient_of_loss(
                y_predicted=predicted, y_actual=y_train, loss_type=self.loss_type
            )
            classifier = generate_stump(
                x_train=x_train,
                y_train=negative_grad_loss_vector,
                loss_type=self.loss_type,
            )
            self.classifiers.append(classifier)
            self.learning_rates.append(self.learning_rate)

    def view_table(self, x: npt.NDArray, y: npt.NDArray) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        console = Console()

        for i, classifier in enumerate(self.classifiers):
            # Get predictions for this step and cumulative predictions
            y_pred_step = np.array([classifier(xi) for xi in x])
            y_pred_total = self.predict(x, depth=i + 1)

            # Calculate residuals and pseudo-residuals
            residuals = y_pred_total - y
            squared_losses = residuals**2

            if self.loss_type == "squared":
                pseudo_residuals = -2 * residuals
                resid_col_name = "-∇(Loss) = -(ỹ - y)"
            else:  # loss_type == "absolute"
                pseudo_residuals = -np.sign(residuals)
                resid_col_name = "-∇(Loss) = -sign(ỹ - y)"

            # Create and populate table
            table = Table(title=f"Classifier {i}")
            columns = [
                "x",
                "y_actual",
                f"y_pred h({i})",
                "y_pred_total (ỹ)",
                "Residual (ỹ - y)",
                "Squared loss (ỹ - y)²/2",
                resid_col_name,
            ]

            for col in columns:
                table.add_column(col, justify="right")

            for xi, yi, pred_i, pred_total, resid, sq_loss, pseudo_resid in zip(
                x,
                y,
                y_pred_step,
                y_pred_total,
                residuals,
                squared_losses,
                pseudo_residuals,
            ):
                table.add_row(
                    f"{xi:.4f}",
                    f"{yi:.4f}",
                    f"{pred_i:.2f}",
                    f"{pred_total:.4f}",
                    f"{resid:.2f}",
                    f"{sq_loss:.3f}",
                    f"{pseudo_resid:.2f}",
                )

            console.print(table)

    def view_graph(self, x: npt.NDArray, y: npt.NDArray):
        x = np.asarray(x)
        y = np.asarray(y)

        losses = []
        for n_iterations in range(self.n_iterations):
            predictions: npt.NDArray = self.predict(x, depth=n_iterations)
            losses.append(np.mean(compute_loss(y_predicted=predictions, y_actual=y, loss_type=self.loss_type)))

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(losses)), losses, marker="o", linestyle="-", color="b")
        plt.title("Prediction Loss vs Number of Iterations")
        plt.xlabel("Number of Iterations")
        plt.ylabel(f"{self.loss_type.title()} Loss")
        plt.grid(True)
        plt.show()


def generate_dataset(size: int, testing_frac: float) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    rng = np.random.default_rng()
    rng = np.random.default_rng(seed=123)

    n: int = size
    x: npt.NDArray = rng.uniform(0, 1, n)
    y: npt.NDArray = np.sin(2 * np.pi * x)
    y += rng.normal(0, 0.01, n)

    frac_for_testing: float = 0.2
    split_idx: int = int((1 - frac_for_testing) * n)
    x_train: npt.NDArray = x[:split_idx]
    y_train: npt.NDArray = y[:split_idx]

    x_test: npt.NDArray = x[split_idx:]
    y_test: npt.NDArray = y[split_idx:]

    return x_train, y_train, x_test, y_test


def simple_example():
    loss_type = "absolute"
    x_train = np.array([0, 1, 4, 7, 11, 14])
    y_train = np.array([0, 1, 16, 49, 121, 196])
    gb = GradientBoost(n_iterations=50, loss_type=loss_type, learning_rate=1)
    gb.train(x_train=x_train, y_train=y_train)
    gb.view_table(x=x_train, y=y_train)
    gb.view_graph(x=x_train, y=y_train)


def main():
    loss_type = "squared"
    x_train, y_train, x_test, y_test = generate_dataset(50, 0.3)
    gb = GradientBoost(n_iterations=20, loss_type=loss_type, learning_rate=0.1)
    gb.train(x_train=x_train, y_train=y_train)
    gb.view_table(x=x_train, y=y_train)
    gb.view_graph(x=x_train, y=y_train)


if __name__ == "__main__":
    main()
