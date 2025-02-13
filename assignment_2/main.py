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

    def collect_indices_of_digit(self, labels, digit: int):
        digit_indices = np.where(labels == digit)[0]
        print(f"Total training images of digit {digit}: {len(digit_indices)}")
        return digit_indices


    def random_seive(self, count : int, training_data : np.ndarray, training_labels : np.ndarray):
        training_data_dict = dict()
        for digit in range(0, 3):
            indices = self.collect_indices_of_digit(training_labels, digit)
            np.random.choice(uuin
            training_data_dict[digit] =


    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        for i in range(len(y_train)):
            print(f"{y_train[i]}: {y_train[i]}")
        return (x_train, y_train), (x_test, y_test)


def init() -> tuple:
    # Set file paths
    input_path = 'input'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    # Load MNIST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
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




def pca():
    pass


def calculate_mle(image_indices, images):
    # Create a flattened, normalized np array of images
    raw_data = []

    mean_temp = np.zeros(784)
    print(f"shape of mean_temp = {mean_temp.shape}")

    for i in image_indices:
        f = images[i].flatten(order='F')  # Flattening an image matrix to an array, Fortran-like, Column-stacking
        print(f"shape of f = {f.shape}")
        mean_temp += f
        f = (f - f.min()) / (f.max() - f.min())  # Normalization, to make data entries between 0 and 1
        raw_data.append(f)
    flattened_images = np.array(raw_data)  # Our normalized image vectors

    mean_temp /= len(image_indices)
    mean_temp = mean_temp.reshape(images[0].shape, order='F')
    print(f"shape of mean_temp = {mean_temp.shape}")
    # show_images([mean_temp], ["MeanTitle"])

    n = len(image_indices)  # Number of training images of a particular dataset
    assert n == len(flattened_images)

    # Mean Calculation
    mean = np.sum(flattened_images, axis=0)  # Get the vector sum
    mean = np.divide(mean, n)  # Divide it by the number of images

    # Covariance Calculation
    covariance = np.zeros((len(mean), len(mean)))  # Initialize cov matrix to a 0-matrix
    for x in flattened_images:
        covariance += ((x - mean) * ((x - mean).transpose()))
    covariance /= n  # Divide by the number of samples

    print(f"Covariance matrix: {covariance}")
    print(f"Sum of covariance matrix: {np.sum(covariance)}")

    rng = np.random.default_rng()
    vector = rng.multivariate_normal(mean, covariance, 1).T
    print(f"shape of vector = {vector.shape}")
    vector = vector.reshape(images[0].shape, order='F')
    print(f"shape of vector = {vector.shape}")

    show_images([vector], title_texts=['Sampled image'])


def main():
    # x are images and y are labels
    (x_train, y_train), (x_test, y_test) = init()
    indices = dict()
    for digit in range(0, 10):
        indices[digit] = collect_images_of_digit(y_train, digit)
    calculate_mle(indices[8], x_train)


if __name__ == '__main__':
    main()
