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

    data_matrix = np.concatenate(list(training_data.values()), axis=0) #X
    mean = np.mean(data_matrix, axis=0) # mu along each column
    centered_data_matrix = data_matrix - mean
    num_samples = data_matrix.shape[0]
    covariance_matrix = np.matmul(centered_data_matrix, centered_data_matrix.T) / (num_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    tupled_eigendata = zip(eigenvalues, eigenvectors.T)
    sorted_eigendata = sorted(tupled_eigendata, key=lambda x: x[0], reverse=True)
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)

    pca_components = 0
    for i in range(len(eigenvalues) + 1):
        variance = sum(sorted_eigenvalues[:i]) / sum(sorted_eigenvalues)
        print(f"PCA with {i} components has variance {variance}")
        if variance >= desired_variance:
            pca_components = i
            print(f"Reached/Surpassed desired variance of {desired_variance} with {pca_components} components")
            break

def fda(training_data : dict[int: np.ndarray]):
    all_matrices = np.concatenate(list(training_data.values()), axis=0)
    overall_mean = np.mean(all_matrices, axis=0)

    between_class_scatter = np.zeros((28*28, 28*28))
    num_classes = len(training_data.keys())
    for c in training_data.keys():
        num_samples_of_class = len(training_data[c])
        class_mean = np.mean(training_data[c], axis=0)
        outer_product = np.outer(class_mean - overall_mean, class_mean - overall_mean)
        between_class_scatter += num_samples_of_class * outer_product

    within_class_scatter = np.zeros((28*28, 28*28))
    for c in training_data.keys():
        class_mean = np.mean(training_data[c], axis=0)
        class_matrix = sum([np.outer(x - class_mean, x - class_mean) for x in training_data[c]])
        within_class_scatter += class_matrix

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(within_class_scatter) @ between_class_scatter)
    indices = np.argsort(-eigenvalues.real)
    W = eigenvectors[:, indices]



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
        covariance += ((x - mean) * ((x - mean).transpose()))
    covariance /= n  # Divide by the number of samples


def main():
    # x are images and y are labels
    training_data, test_data = init()
    print(training_data)
    print(f"Len of training_data: {len(training_data)}")
    print(f"Len of training_data[1]: {len(training_data[1])}")
    print(f"Shape of training_data[1] : {training_data[1].shape}")
    print(f"Shape of training_data[1][0] : {training_data[1][0].shape}")
    flattened_training_data = dict()
    flatten_dict(training_data, flattened_training_data)

    for v in training_data.values():
        print(f"{v.shape}")
    # calculate_mle(2, training_data)
    mle(1, flattened_training_data)
    pca(flattened_training_data, 0.95)
    fda(flattened_training_data)


if __name__ == '__main__':
    main()
