from matplotlib import pyplot as plt
import numpy as np
from typing import Final, Literal, Tuple
from idx2numpy import convert_from_file as load_idx_as_numpy
from sklearn.decomposition import PCA


# Decision Stump and AdaBoost classes from previous implementation
class DecisionStump:
    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float | None = None,
        polarity: Literal[-1, 1] = 1,
    ):
        self.feature_idx = feature_idx
        self.threshold: float | None = threshold
        self.polarity: Literal[-1, 1] = polarity
        self.weight: float = 0.0
        self.error: float = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        # Apply the decision rule based on feature, threshold, and polarity
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] >= self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_estimators: int = 200):
        self.n_estimators: Final[int] = n_estimators
        self.stumps: list[DecisionStump] = []
        self.train_error: list[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n_samples = x.shape[0]

        # Initialize sample weights to 1/N
        sample_weights = np.full(n_samples, fill_value=1 / n_samples)

        # Train weak classifiers
        for i in range(self.n_estimators):
            # Train a weak classifier (decision stump)
            stump, error, predictions = self.create_stump(x, y, sample_weights)

            # Avoid division by zero or log(0)
            epsilon = 1e-10
            error = np.clip(error, epsilon, 1 - epsilon)
            # stump.weight = 0.5 * np.log((1 - error) / error)
            stump.weight = 1

            # Update sample weights
            sample_weights = self.update_weights(sample_weights, stump.weight, y, predictions)

            # Add the classifier to ensemble
            self.stumps.append(stump)

    def create_stump(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> Tuple[DecisionStump, float, np.ndarray]:
        n_features = x.shape[1]
        best_error = float("inf")
        best_stump = DecisionStump()
        best_predictions = None

        # For each feature
        for feature_idx in range(n_features):
            # Get unique values for the feature
            feature_values = np.sort(np.unique(x[:, feature_idx]))

            # Calculate thresholds as midpoints between consecutive values
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            # If there's only one unique value, add a small offset to create a threshold
            if len(thresholds) == 0:
                thresholds = [feature_values[0] - 0.1]

            # For each threshold
            for threshold in thresholds:
                # Try both polarities
                for polarity in (1, -1):
                    stump = DecisionStump(feature_idx, threshold, polarity)

                    # Make predictions
                    predictions = stump.predict(x)

                    # Calculate weighted error
                    error = np.sum(weights * (predictions != y))

                    # Update if this is the best stump so far
                    if error < best_error:
                        best_error = error
                        best_stump = stump
                        best_predictions = predictions

        best_stump.error = best_error
        return best_stump, best_error, best_predictions

    def update_weights(
        self, weights: np.ndarray, stump_weight: float, y: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        # Update weights
        weights = weights * np.exp(-stump_weight * y * predictions)

        # Normalize weights
        weights = weights / np.sum(weights)

        return weights

    def predict(self, x: np.ndarray, n_rounds: int | None = None) -> np.ndarray:
        n_samples = x.shape[0]  # x is (n_samples, n_features)

        # Sum up weighted predictions from all weak classifiers
        ensemble_predictions = np.zeros(n_samples)  # (n_samples,) with values in {-1, 1}

        for stump in self.stumps[:n_rounds]:
            predictions = stump.predict(x)
            ensemble_predictions += stump.weight * predictions

        # Return sign of the ensemble predictions
        return np.sign(ensemble_predictions)


def load_dataset():
    print("Loading MNIST dataset with idx2numpy...")

    # Load the data using idx2numpy
    x_train: np.ndarray = load_idx_as_numpy("train-images.idx3-ubyte").astype(np.float64)
    y_train: np.ndarray = load_idx_as_numpy("train-labels.idx1-ubyte")
    x_test: np.ndarray = load_idx_as_numpy("t10k-images.idx3-ubyte").astype(np.float64)
    y_test: np.ndarray = load_idx_as_numpy("t10k-labels.idx1-ubyte")

    print(f"Original data shapes: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Original data shapes: x_test={x_test.shape}, y_test={y_test.shape}")

    # Flatten the images from (num_samples, 28, 28) → (num_samples, 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    print(f"Flattened data shapes: x_train={x_train.shape}, x_test={x_test.shape}")

    # Filter only digits 0 and 1
    train_filter = np.logical_or(y_train == 0, y_train == 1)
    test_filter = np.logical_or(y_test == 0, y_test == 1)

    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]
    x_test_filtered = x_test[test_filter]
    y_test_filtered = y_test[test_filter]

    # Keep only 1000 samples for training
    k = 1000
    x_train_filtered = x_train_filtered[:k]
    y_train_filtered = y_train_filtered[:k]

    # Convert labels to -1 and +1 for AdaBoost
    y_train_adaboost = np.where(y_train_filtered == 0, -1, 1)
    y_test_adaboost = np.where(y_test_filtered == 0, -1, 1)

    print(f"Filtered data shapes: x_train={x_train_filtered.shape}, y_train={y_train_filtered.shape}")
    print(f"Filtered data shapes: x_test={x_test_filtered.shape}, y_test={y_test_filtered.shape}")
    print(f"Class distribution in training set: {np.bincount(y_train_filtered)}")
    print(f"Class distribution in test set: {np.bincount(y_test_filtered)}")

    return x_train_filtered, y_train_adaboost, x_test_filtered, y_test_adaboost


def compute_error_vs_rounds(x: np.ndarray, y: np.ndarray, model: AdaBoost, title: str):
    errors = []
    for i in range(model.n_estimators):
        y_pred = model.predict(x, i)
        error = np.mean(np.sign(y_pred) != y)  # Compute error rate
        errors.append(error)

    # Plotting using fig and ax
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(errors) + 1), errors, marker="o", label=f"{title}")
    ax.set_title(f"{title} vs Boosting Rounds")
    ax.set_xlabel("Number of Boosting Rounds")
    ax.set_ylabel("Error Rate")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    x_train, y_train, x_test, y_test = load_dataset()

    # Normalize pixel values to [0, 1] and Apply PCA to reduce dimensions to 5
    print("Applying PCA to reduce dimensions to 5...")
    pca = PCA(n_components=5)
    x_train = pca.fit_transform(x_train / 255.0)
    x_test = pca.transform(x_test / 255.0)
    print(f"Reduced data shapes: x_train={x_train.shape}, x_test={x_test.shape}")

    x_train_mean = x_train.mean(axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    # Train AdaBoost on the reduced dimensionality data
    print("\nTraining AdaBoost classifier...")
    model = AdaBoost(n_estimators=20)
    model.fit(x_train, y_train)

    compute_error_vs_rounds(x_train, y_train, model, "Training Error")
    compute_error_vs_rounds(x_test, y_test, model, "Test Error")


if __name__ == "__main__":
    main()
