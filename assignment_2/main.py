import numpy as np
import struct
from os.path import join
from typing import Tuple
import matplotlib.pyplot as plt
import idx2numpy


class MnistDataloader:
    def __init__(self, training_images_filepath: str, training_labels_filepath: str,
                 test_images_filepath: str, test_labels_filepath: str):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def load_data(self) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        x_train = idx2numpy.convert_from_file(self.training_images_filepath)
        y_train = idx2numpy.convert_from_file(self.training_labels_filepath)
        x_test = idx2numpy.convert_from_file(self.test_images_filepath)
        y_test = idx2numpy.convert_from_file(self.test_labels_filepath)

        # Refine data into dictionary format
        training_data = self._refine_data(x_train, y_train, sieve=True, count=100)
        test_data = self._refine_data(x_test, y_test, sieve=False)

        return training_data, test_data

    def _refine_data(self, images: np.ndarray, labels: np.ndarray, sieve: bool, count: int = -1) -> dict[int, np.ndarray]:
        data_dict = {}
        for digit in range(3):  # Only consider classes 0, 1, 2
            indices = np.where(labels == digit)[0]
            if sieve:
                selected_indices = np.random.choice(indices, count, replace=False)
                data_dict[digit] = images[selected_indices]
            else:
                data_dict[digit] = images[indices]
        return data_dict


def init() -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    input_path = 'input'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MNIST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)
    return mnist_dataloader.load_data()


def show_images(images, title_texts) -> None:
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(30, 20))
    for index, (image, title_text) in enumerate(zip(images, title_texts)):
        plt.subplot(rows, cols, index + 1)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text:
            plt.title(title_text, fontsize=15)
    plt.tight_layout()
    plt.show()


def flatten_and_normalize_solo(image: np.ndarray) -> np.ndarray:
    f = image.flatten(order='F').astype(np.float64)  # Convert to float to avoid dtype issues
    norm = np.linalg.norm(f)  # Compute the norm

    if norm == 0:
        return f  # Avoid division by zero, return as is

    f /= norm  # Normalize the vector
    return f


def flatten_and_normalize(dataset: np.ndarray) -> np.ndarray:
    return np.array([flatten_and_normalize_solo(x) for x in dataset])


def flatten_dict(old_dict: dict[int: np.ndarray], new_dict: dict) -> None:
    for key in old_dict.keys():
        new_dict[key] = flatten_and_normalize(old_dict[key])


def lda_test(testing_data: dict[int: np.ndarray], class_means: dict[int: np.ndarray],
             covariance_matrix: np.ndarray) -> float:
    data_matrix = np.concatenate(list(testing_data.values()), axis=0)
    classes = testing_data.keys()
    labels = np.concatenate([[c] * len(testing_data[c]) for c in classes], axis=0)

    saved_w = dict()  # For memoization
    saved_w_0 = dict()  # For memoization

    # Regularize the covariance matrix to ensure it's invertible
    # reg_param = 1e-6
    # sigma_reg = covariance_matrix + reg_param * np.eye(covariance_matrix.shape[0])
    sigma_inverse = np.linalg.pinv(
        covariance_matrix)  # Using regular inverse instead of pseudo-inverse since already regularized

    def lda_g_i(x: np.ndarray, i: int, mu_i: np.ndarray, sigma: np.ndarray) -> float:
        """
        Using formula of Duda (2.6.2)
        LDA considers covariance to be the same for all classes.
        So the quadratic term in x vanishes
        """
        if i not in saved_w:
            w_i = sigma_inverse @ mu_i
            w_i0 = -0.5 * (mu_i.T @ sigma_inverse @ mu_i)  # + lnP(omega_i)
            # skipping + lnP(omega_i) because we are assuming it to be equal for all i
            # as the prior is equal because number of samples of all digits are equal = 100
            saved_w[i], saved_w_0[i] = w_i, w_i0  # caching the values

        w_i, w_i0 = saved_w[i], saved_w_0[i]
        g = (w_i.T @ x) + w_i0
        return g

    output = []
    for x in data_matrix:
        g_is = [lda_g_i(x, i, class_means[i], covariance_matrix) for i in classes]
        prediction = np.argmax(g_is, axis=0)
        output.append(prediction)

    correct = 0
    total = len(output)
    for i in range(total):
        if output[i] == labels[i]:
            correct += 1
    accuracy = correct / total
    return accuracy

    # output = np.array(output)
    # accuracy = np.mean(output == labels)
    # return accuracy


def lda_train(training_data: dict[int: np.ndarray]) -> Tuple[dict[int: np.ndarray], np.ndarray]:
    feature_dim = training_data[0].shape[1]
    total_samples = sum(len(training_data[c]) for c in training_data.keys())
    classes = training_data.keys()

    class_means = {c: np.mean(training_data[c], axis=0) for c in classes}
    covariance_matrix = np.zeros((feature_dim, feature_dim))
    for c in classes:
        covariance_matrix += np.matmul((training_data[c] - class_means[c]).T, (training_data[c] - class_means[c]))
    covariance_matrix /= (total_samples - 1)

    return class_means, covariance_matrix


def qda_test(testing_data: dict[int, np.ndarray], class_means: dict[int, np.ndarray],
             class_covariances: dict[int, np.ndarray]) -> float:
    data_matrix = np.concatenate(list(testing_data.values()), axis=0)  # Shape: (N, 784)
    classes = list(testing_data.keys())
    labels = np.concatenate([[c] * len(testing_data[c]) for c in classes], axis=0)  # Shape: (N,)

    # Memoization dictionaries
    saved_W = {}  # Stores W_i for each class
    saved_w = {}  # Stores w_i for each class
    saved_w_0 = {}  # Stores w_i0 for each class

    def qda_g_i(x: np.ndarray, i: int, mu_i: np.ndarray, sigma_i: np.ndarray) -> float:
        """
        Using formula of Duda (2.6.3)
        QDA considers covariance for each class to be different
        So we'll have to deal with the quadratic term in x as well.

        x: Shape: (784,)
        i: Class index.
        mu_i: Shape: (784,)
        sigma_i: Shape: (784, 784)
        """
        if i not in saved_W:
            # Regularize the covariance matrix to ensure it's invertible
            # reg_param = 1e-6
            # sigma_i_reg = sigma_i + reg_param * np.eye(sigma_i.shape[0])

            # Compute and cache terms if not already cached
            sigma_i_inverse = np.linalg.pinv(
                sigma_i)  # Shape: (784, 784) # Taking regular inverse as matrix regularized, sigma_reg if regularized
            W_i = -0.5 * sigma_i_inverse  # Shape: (784, 784)
            w_i = sigma_i_inverse @ mu_i  # Shape: (784, 1)
            # w_i0 = (-0.5 * (mu_i.T @ sigma_i_inverse @ mu_i)) + (-0.5 * np.linalg.slogdet(sigma_i_reg)[1])  # Scalar
            w_i0 = (-0.5 * (mu_i.T @ sigma_i_inverse @ mu_i)) + (-0.5 * np.linalg.slogdet(sigma_i)[1])  # Scalar

            # Cache the computed values
            saved_W[i], saved_w[i], saved_w_0[i] = W_i, w_i, w_i0

        # Skipping + lnP(omega_i) because we are assuming it to be equal for all i
        # as the prior is equal because number of samples of all digits are equal = 100

        # Retrieve cached values
        W_i, w_i, w_i0 = saved_W[i], saved_w[i], saved_w_0[i]

        # Reshape x to a column vector
        x = x[:, np.newaxis]  # Shape: (784, 1)

        # Compute discriminant value
        g = x.T @ W_i @ x + w_i.T @ x + w_i0
        return g.item()  # Convert to scalar

    # Compute predictions
    output = []
    for x in data_matrix:
        g_is = [qda_g_i(x, i, class_means[i], class_covariances[i]) for i in classes]
        prediction = np.argmax(g_is, axis=0)
        output.append(prediction)

    # Compute accuracy
    output = np.array(output)
    accuracy = np.mean(output == labels)
    return accuracy


def qda_train(training_data: dict[int, np.ndarray]) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    classes = training_data.keys()
    num_samples = {c: len(training_data[c]) for c in classes}

    # Compute class means
    class_means = {c: np.mean(training_data[c], axis=0) for c in classes}

    # Compute class covariance matrices
    class_covariances = {}
    for c in classes:
        centered_data = training_data[c] - class_means[c]  # Shape: (n_samples, n_features)
        # Compute covariance matrix
        covariance_matrix = (centered_data.T @ centered_data) / (num_samples[c] - 1)  # Shape: (n_features, n_features)
        class_covariances[c] = covariance_matrix

    return class_means, class_covariances


def mle(digit: int, training_data: dict[int: np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Create a flattened, normalized np array of images
    images = training_data[digit]

    n = len(images)  # Number of training images of a particular dataset

    # Mean Calculation
    mean = np.sum(images, axis=0)  # Get the vector sum
    mean = np.divide(mean, n)  # Divide it by the number of images

    # Covariance Calculation
    covariance = np.zeros((len(mean), len(mean)))  # Initialize cov matrix to a 0-matrix
    for x in images:
        covariance += np.matmul((x - mean), (x - mean).T)
    covariance /= n  # Divide by the number of samples

    return mean, covariance


def pca(training_data: dict[int: np.ndarray], desired_variance: float) -> Tuple[np.ndarray, np.ndarray]:
    assert 1 >= desired_variance >= 0

    # Concatenate all training data into a single matrix
    data_matrix = np.concatenate(list(training_data.values()), axis=0)  # Shape: (num_samples, 784)

    # Compute mean and center the data
    mean = np.mean(data_matrix, axis=0)  # Shape: (784,)
    centered_data_matrix = data_matrix - mean  # Shape: (num_samples, 784)

    # Compute covariance matrix (X^T X / (n-1))
    num_samples = data_matrix.shape[0]
    covariance_matrix = (centered_data_matrix.T @ centered_data_matrix) / (num_samples - 1)  # Shape: (784, 784)

    # Compute eigenvalues and eigenvectors using eigh (for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # Guarantees real eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)  # Explicitly take the real part

    # Sort eigenvalues and eigenvectors in descending order
    indices = np.argsort(-eigenvalues)
    sorted_eigenvalues = eigenvalues[indices]
    sorted_eigenvectors = eigenvectors[:, indices]

    # Finding the number of PCA components required to gain the desired variance
    pca_components = 0
    for i in range(len(eigenvalues) + 1):
        variance = sum(sorted_eigenvalues[:i]) / sum(sorted_eigenvalues)
        # print(f"PCA with {i} components has variance {variance}")
        # TODO : Can use binary search here, lower bound = 0, upper bound = len(eigenvalues)
        if variance >= desired_variance:
            pca_components = i+1
            print(f"[INFO] Reached/Surpassed desired variance of {desired_variance} with {pca_components} components")
            break


    # Constructing U_p and Y matrices
    U_p = sorted_eigenvectors[:, :pca_components]
    return U_p, mean


def fda(training_data: dict[int: np.ndarray]) -> np.ndarray:  # Returns W
    data_matrix = np.concatenate(list(training_data.values()), axis=0)
    overall_mean = np.mean(data_matrix, axis=0)
    feature_dim = overall_mean.shape[0]

    between_class_scatter = np.zeros((feature_dim, feature_dim))
    num_classes = len(training_data.keys())
    for c in training_data.keys():
        num_samples_of_class = len(training_data[c])
        class_mean = np.mean(training_data[c], axis=0)
        outer_product = np.outer(class_mean - overall_mean, class_mean - overall_mean)
        between_class_scatter += (num_samples_of_class * outer_product)

    within_class_scatter = np.zeros((feature_dim, feature_dim))
    for c in training_data.keys():
        class_mean = np.mean(training_data[c], axis=0)
        class_matrix = sum([np.outer(x - class_mean, x - class_mean) for x in training_data[c]])
        within_class_scatter += class_matrix

    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(within_class_scatter) @ between_class_scatter)
    print(f"Converting to reals!")
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    # Sorting by eigenvalues in descending order
    indices = np.argsort(-eigenvalues.real)
    # Eigenvectors corresponding to the eigenvalues in descending order are in W
    W = eigenvectors[:, indices]
    return W

def do_mle(training_data: dict[int: np.ndarray]) -> None:
    print("\n\n[INFO] Testing MLE...")

    for c in range(3):
        mean, covariance = mle(c, training_data)
        print(f"[SUCCESS ({c + 1}/{3})] Calculated mean and covariance for class {c}")
        print(f"[INFO] Mean for class {c} is: {mean}")
        print(f"[INFO] Covariance for class {c} is: {covariance}")

def plot_transformed_data(transformed_data: dict[int: np.ndarray], title: str):
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    markers = ['o', 's', 'D']
    for c in transformed_data.keys():
        plt.scatter(transformed_data[c][:, 0], transformed_data[c][:, 1], c=colors[c], marker=markers[c], label=f'Class {c}')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

def test_pca(training_data: dict[int: np.ndarray], test_data: dict[int: np.ndarray], desired_variance: float) -> None:
    print("\n[INFO] Testing PCA...")
    U_p, mu = pca(training_data, desired_variance)
    U_p = U_p[:, :2] # for our two component case
    print(f"Shape of U_p is {U_p.shape}")
    print(f"[SUCCESS (1/8)] Calculated U_p and mu")

    pca_training_data = {c: np.matmul(data - mu, U_p) for c, data in training_data.items()}
    pca_test_data = {c: np.matmul(data - mu, U_p) for c, data in test_data.items()}
    print(f"[SUCCESS (2/8)] Projected training and test data")

    class_means, covariance_matrix = lda_train(pca_training_data)
    print(f"[SUCCESS (3/8)] Class means and Common Covariance Matrix obtained by doing LDA on training data")

    accuracy = lda_test(pca_training_data, class_means, covariance_matrix)
    print(f"[SUCCESS (4/8)] Used the above mu_i's and sigma to run LDA on training data")
    print(f"[INFO] LDA's accuracy after PCA came out to be {accuracy:.4f}")

    accuracy = lda_test(pca_test_data, class_means, covariance_matrix)
    print(f"[SUCCESS (5/8)] Used the above mu_i's and sigma to run LDA on test data")
    print(f"[INFO] LDA's accuracy after PCA came out to be {accuracy:.4f}")

    class_means, class_covariances = qda_train(pca_training_data)
    print(f"[SUCCESS (6/8)] Class means and Class Covariance Matrices obtained by doing QDA on training data")

    accuracy = qda_test(pca_training_data, class_means, class_covariances)
    print(f"[SUCCESS (7/8)] Used the above mu_i's and sigma_i's to run QDA on training data")
    print(f"[INFO] QDA's accuracy after PCA came out to be {accuracy:.4f}")

    accuracy = qda_test(pca_test_data, class_means, class_covariances)
    print(f"[SUCCESS (8/8)] Used the above mu_i's and sigma_i's to run QDA on test data")
    print(f"[INFO] QDA's accuracy after PCA came out to be {accuracy:.4f}")


def test_fda(training_data: dict[int: np.ndarray], test_data: dict[int: np.ndarray]) -> None:
    print("\n\n[INFO] Testing FDA...")

    W = fda(training_data)
    print(f"[SUCCESS (1/8)] Calculated W")

    fda_training_data = {c: np.matmul(data, W) for c, data in training_data.items()}
    fda_test_data = {c: np.matmul(data, W) for c, data in test_data.items()}
    print(f"[SUCCESS (2/8)] Projected training and test data")

    class_means, covariance_matrix = lda_train(fda_training_data)
    print(f"[SUCCESS (3/8)] Class means and Common Covariance Matrix obtained by doing LDA on training data")

    accuracy = lda_test(fda_training_data, class_means, covariance_matrix)
    print(f"[SUCCESS (4/8)] Used the above mu_i's and sigma to run LDA on training data")
    print(f"[INFO] LDA's accuracy after FDA came out to be {accuracy:.4f}")

    accuracy = lda_test(fda_test_data, class_means, covariance_matrix)
    print(f"[SUCCESS (5/8)] Used the above mu_i's and sigma to run LDA on test data")
    print(f"[INFO] LDA's accuracy after FDA came out to be {accuracy:.4f}")

    class_means, class_covariances = qda_train(fda_training_data)
    print(f"[SUCCESS (6/8)] Class means and Class Covariance Matrices obtained by doing QDA on training data")

    accuracy = qda_test(fda_training_data, class_means, class_covariances)
    print(f"[SUCCESS (7/8)] Used the above mu_i's and sigma_i's to run QDA on training data")
    print(f"[INFO] QDA's accuracy after FDA came out to be {accuracy:.4f}")

    accuracy = qda_test(fda_test_data, class_means, class_covariances)
    print(f"[SUCCESS (8/8)] Used the above mu_i's and sigma_i's to run QDA on test data")
    print(f"[INFO] QDA's accuracy after FDA came out to be {accuracy:.4f}")


def main():
    # x are images and y are labels
    training_data, test_data = init()
    flattened_training_data = dict()
    flatten_dict(training_data, flattened_training_data)
    flattened_test_data = dict()
    flatten_dict(test_data, flattened_test_data)

    # do_mle(flattened_training_data)
    test_pca(flattened_training_data, flattened_test_data, 1)
    # test_fda(flattened_training_data, flattened_test_data)


if __name__ == '__main__':
    main()
