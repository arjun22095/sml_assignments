import math
from typing import Literal, NamedTuple

import numpy as np

rng = np.random.default_rng(seed=123)


def sigmoid(x: float):
    return 1 / (1 + math.exp(-x))


def mse(a: float, b: float) -> float:
    return ((a - b) ** 2) / 2


def compute_label(a: float) -> Literal[0, 1]:
    return 1 if a >= 0 else 0


class Dataset(NamedTuple):
    x: list[float]
    y: list[int]

    def __iter__(self):
        yield from zip(self.x, self.y)


class NeuralNetwork:
    def __init__(self, eta: float = 0.1):
        self.eta = eta

        # Input -> Hidden
        self.w0_0: float = rng.uniform()
        self.w0_1: float = rng.uniform()
        self.b0_0: float = rng.uniform()

        # Hidden -> Output
        self.w1_0: float = rng.uniform()
        self.b1_0: float = rng.uniform()

        # Activations
        self.a0_0: float = 0.0
        self.a1_0: float = 0.0

    def forward(self, x: tuple[float, float]) -> float:
        self.a0_0 = sigmoid(self.w0_0 * x[0] + self.w0_1 * x[1] + self.b0_0)
        self.a1_0 = self.w1_0 * self.a0_0 + self.b1_0
        return self.a1_0

    __call__ = forward

    def learn(self, x: tuple[float, float], y: Literal[0, 1]):
        y_pred = self.forward(x)
        # Loss = 1/2n SUM (y_pred - y)^2
        # Loss = 1/2n SUM (a1_0 - y)^2
        # Loss = 1/2n SUM ((w1_0 * a0_0 + b1_0) - y)^2
        # Loss = 1/2n SUM ((w1_0 * sigmoid(w0_0 * x0 + w0_1 * x1 + b0_0) + b1_0) - y)^2

        # Loss = 1/2 ((w1_0 * sigmoid(w0_0 * x0 + w0_1 * x1 + b0_0) + b1_0) - y)^2

        # Hidden <- Output
        dL_dw1_0 = (y_pred - y) * self.a0_0
        dL_db1_0 = (y_pred - y) * 1

        # Input <- Hidden
        dL_dw0_1 = dL_dw1_0 * (self.a0_0 * (1 - self.a0_0)) * x[1]
        dL_dw0_0 = dL_dw1_0 * (self.a0_0 * (1 - self.a0_0)) * x[0]
        dL_db0_0 = dL_dw1_0 * (self.a0_0 * (1 - self.a0_0)) * 1

        # Gradient descent
        self.w1_0 -= self.eta * dL_dw1_0
        self.b1_0 -= self.eta * dL_db1_0

        self.w0_1 -= self.eta * dL_dw0_1
        self.w0_0 -= self.eta * dL_dw0_0
        self.b0_0 -= self.eta * dL_db0_0


def generate_dataset(n: int = 10):
    x0 = rng.normal([-1, -1], size=(n, 2))
    y0 = np.full(len(x0), 0, dtype=np.int8)

    x1 = rng.normal([1, 1], size=(n, 2))
    y1 = np.full(len(x1), 1, dtype=np.int8)

    x = np.concatenate((x0, x1))
    y = np.concatenate((y0, y1))

    p = rng.permutation(len(x))
    x = x[p]
    y = y[p]

    train_set = Dataset(x=x[:n], y=y[:n])
    test_set = Dataset(x=x[n:], y=y[n:])

    return train_set, test_set


def evaluate(
    net: NeuralNetwork,
    dataset: Dataset,
    loss_fn,
    train: bool = False,
):
    total_loss = 0.0
    n_correct = 0
    for x, y in dataset:
        y_pred = net(x)
        total_loss += loss_fn(y_pred, y)

        label_pred = compute_label(y_pred)
        n_correct += label_pred == y

        if not train:
            print(f"y = {y} | y_pred = {y_pred}, {label_pred=}")

        if train:
            net.learn(x, y)

    accuracy = n_correct / len(dataset.x)

    label = "Train" if train else "Test"
    print(f"{label} MSE loss: {total_loss:.3f}")
    print(f"{label} Accuracy: {100 * accuracy:.3f}\n")


def main():
    train_set, test_set = generate_dataset(n=10)
    net = NeuralNetwork(eta=0.1)
    evaluate(net, train_set, loss_fn=mse, train=True)
    evaluate(net, test_set, loss_fn=mse, train=False)


if __name__ == "__main__":
    main()
