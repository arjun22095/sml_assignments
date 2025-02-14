import numpy as np
import struct
from os.path import join
from typing import Tuple
import matplotlib.pyplot as plt


class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch for labels, expected 2049, got {magic}')
            labels = np.frombuffer(file.read(), dtype=np.uint8)  # Efficiently read as np array

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch for images, expected 2051, got {magic}')
            images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows, cols)

        return images, labels


    def refine_data(self, to_sieve : bool, data: np.ndarray, labels: np.ndarray, count: int = -1) -> dict[int: np.ndarray]:
        data_dict = dict()
        for digit in range(0, 3):
            indices = np.where(labels == digit)[0]
            if to_sieve: # Sieve the data for the training data set
                selected_indices = np.random.choice(indices, count, replace=False)
                data_dict[digit] = data[selected_indices]
            else: # Called for test data set
                data_dict[digit] = data[indices]

        return data_dict


    def load_data(self) -> Tuple[dict[int: np.ndarray], dict[int: np.ndarray]]:
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        sieved_training_data = self.refine_data(to_sieve = True, data=x_train, labels=y_train, count=100)
        refined_test_data = self.refine_data(to_sieve = False, data=x_test, labels=y_test)
        return sieved_training_data, refined_test_data

def init() -> Tuple[dict[int: np.ndarray], dict[int: np.ndarray]]:
    # Set file paths
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
    f = image.flatten(order='F')  # Flattening an image matrix to an array, Fortran-like, Column-stacking
    f = (f - f.min()) / (f.max() - f.min())  # Normalization, to make data entries between 0 and 1
    return f

def flatten_and_normalize(dataset: np.ndarray) -> np.ndarray:
    print(f"Flattening dataset with shape {dataset.shape}")
    return np.array([flatten_and_normalize_solo(x) for x in dataset])

def flatten_dict(old_dict: dict[int: np.ndarray], new_dict: dict) -> None:
    for key in old_dict.keys():
        new_dict[key] = flatten_and_normalize(old_dict[key])

def pca(training_data : dict[int: np.ndarray], desired_variance : float) -> None:
    assert 1 >= desired_variance >= 0

    data_matrix = np.concatenate(list(training_data.values()), axis=0).T # X = 784 x 300

    # Calculating mu and X_c
    mean = np.mean(data_matrix, axis=1) # mu along each row as we transposed the matrix
    centered_data_matrix = data_matrix - mean[:, np.newaxis] # To convert mu from (784,) -> (784,1)
    num_samples = data_matrix.shape[1] # 300

    # Calculating Covariance Matrix
    covariance_matrix = np.matmul(centered_data_matrix, centered_data_matrix.T) / (num_samples - 1)

    # Calculating Eigenvalues & Eigenvectors and sorting them in descending order
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    indices = np.argsort(-eigenvalues)
    sorted_eigenvalues = eigenvalues[indices]
    sorted_eigenvectors = eigenvectors[:, indices]

    # Finding the number of PCA components required to gain the desired variance
    pca_components = 0
    for i in range(len(eigenvalues) + 1):
        variance = sum(sorted_eigenvalues[:i]) / sum(sorted_eigenvalues)
        # print(f"PCA with {i} components has variance {variance}")
        if variance >= desired_variance:
            pca_components = i
            print(f"Reached/Surpassed desired variance of {desired_variance} with {pca_components} components")
            break

    # Constructing U_p and Y matrices
    U_p = sorted_eigenvectors[:, :pca_components]
    print(f"Shape of U_p: {U_p.shape}")
    Y = np.matmul(U_p.T, centered_data_matrix)
    print(f"Shape of Y: {Y.shape}")

def fda(training_data : dict[int: np.ndarray]) -> np.ndarray: # Returns W
    data_matrix = np.concatenate(list(training_data.values()), axis=0)
    overall_mean = np.mean(data_matrix, axis=0)
    feature_dim = overall_mean.shape[0]

    between_class_scatter = np.zeros((feature_dim, feature_dim))
    num_classes = len(training_data.keys())
    for c in training_data.keys():
        num_samples_of_class = len(training_data[c])
        class_mean = np.mean(training_data[c], axis=0)
        outer_product = np.outer(class_mean - overall_mean, class_mean - overall_mean)
        between_class_scatter += num_samples_of_class * outer_product

    within_class_scatter = np.zeros((feature_dim, feature_dim))
    for c in training_data.keys():
        class_mean = np.mean(training_data[c], axis=0)
        class_matrix = sum([np.outer(x - class_mean, x - class_mean) for x in training_data[c]])
        within_class_scatter += class_matrix

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(within_class_scatter) @ between_class_scatter)
    indices = np.argsort(-eigenvalues.real)
    W = eigenvectors[:, indices]
    return W

def mle(digit : int, training_data : dict[int: np.ndarray]):
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

def lda_train(training_data : dict[int: np.ndarray]) -> Tuple[dict[int: np.ndarray], np.ndarray]:
    feature_dim = training_data[0].shape[1]
    total_samples = sum(len(training_data[c]) for c in training_data.keys())
    print(f"Total samples: {total_samples}")
    classes = training_data.keys()
    class_means = {c : np.mean(training_data[c], axis=0) for c in classes}
    covariance_matrix = np.zeros((feature_dim, feature_dim))
    for c in classes:
        covariance_matrix += np.matmul((training_data[c] - class_means[c]).T, (training_data[c] - class_means[c]))
    covariance_matrix /= (len(classes) - 1)

    return class_means, covariance_matrix


def lda_test(testing_data : dict[int: np.ndarray], class_means : dict[int: np.ndarray], covariance_matrix : np.ndarray):
    data_matrix = np.concatenate(list(testing_data.values()), axis=0)
    labels = np.concatenate(list(testing_data.keys()), axis=0)

def main():
    # x are images and y are labels
    training_data, test_data = init()

    flattened_training_data = dict()
    flatten_dict(training_data, flattened_training_data)

    flattened_test_data = dict()
    flatten_dict(test_data, flattened_test_data)

    mle(1, flattened_training_data)
    pca(flattened_training_data, 0.95)
    W = fda(flattened_training_data)

    for c, data in training_data.items():
        print(f"{c}: {data}")

    fda_training_data = {c : np.matmul(data, W) for c, data in flattened_training_data.items()}
    fda_test_data = {c : np.matmul(data, W) for c, data in flattened_test_data.items()}


if __name__ == '__main__':
    main()
