from typing import Dict, Mapping, Optional, Union

import numpy as np
from idx2numpy import convert_from_file as load_idx
from matplotlib import pyplot as plt
from numpy import typing as npt

# B: Batch (e.g., 60000 images)
# D: Dimension (e.g., 784 or 28)


class MnistDataset:
    def __init__(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        class_counts_to_keep: Optional[Mapping[int, Union[int, None]]] = None,
    ) -> None:
        self.x = x  # (B, W, H)
        self.y = y  # (B,)

        if class_counts_to_keep is not None:
            _x, _y = [], []
            for cls, count in class_counts_to_keep.items():
                mask = np.where(self.y == cls)
                _x.append(self.x[mask][:count])
                _y.append(self.y[mask][:count])

            self.x = np.concatenate(_x)
            self.y = np.concatenate(_y)

    def preprocess(self) -> None:
        # Flatten
        self.x = self.x.reshape(self.x.shape[0], -1)  # (B, W, H) -> (B, D)  # D = W * H

        # Normalize
        norms = np.linalg.norm(self.x, keepdims=True, axis=-1)  # (B, 1): Take norm along samples
        self.x = self.x / norms  # (B, D)

        # for x in self.x:  # x: (D,)  # Sanity check
        #     assert np.allclose(x.dot(x), 1), x.dot(x)

    def x_for_class(self, cls: int) -> npt.NDArray[np.float64]:
        mask = np.where(self.y == cls)
        return self.x[mask]

    def __len__(self) -> int:
        return len(self.x)

    def __str__(self) -> str:
        return f"Dataset(x.shape={self.x.shape}, y.shape={self.y.shape})"


def compute_covariance(x: npt.NDArray):
    mean = x.mean(axis=0)  # (B, d) -> (d,)
    x_centered = x - mean  # (B, d)
    assert np.allclose(x_centered.mean(axis=0), 0)  # Along image axis
    return (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)


def mle(train_set: MnistDataset):
    classes: npt.NDArray[np.uint8] = np.unique(train_set.y)
    
    for cls in classes:
        x_cls = train_set.x_for_class(cls)  # (B, D) # B is the number of samples of class i
        cls_mean = x_cls.mean(axis=0)  # (D,) 
        x_centered = x_cls - cls_mean
        cov_cls = (x_centered.T @ x_centered) / (x_cls.shape[0]) # (D, D)
        assert not np.allclose(cov_cls, 0, atol=1e-5)
        np.savetxt(f"mle_cov_class_{cls}.txt", cov_cls, fmt="%.6f")
        np.savetxt(f"mle_mean_class_{cls}.txt", cls_mean, fmt="%.6f")
    

def pca(
    train_set: MnistDataset,
    n_components: Optional[int] = None,
    min_variance: Optional[float] = None,
) -> npt.NDArray[np.float64]:
    assert (n_components is None) ^ (min_variance is None), "Provide either n_components or min_variance"

    # Compute eigenvectors
    eigenresult = np.linalg.eigh(compute_covariance(train_set.x))
    eigenvalues = eigenresult[0]  # (D,)
    eigenvectors = eigenresult[1]  # (D, D) # (D, i-th eigenvector)

    top_eigenvalue_indices = np.argsort(eigenvalues)[::-1]  # (D,) (Descending order)
    top_eigenvalues = eigenvalues[top_eigenvalue_indices]  # (D,)
    top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]  # (D, D) # (D, i-th eigenvector)

    if min_variance is not None:
        assert n_components is None
        assert 0 <= min_variance <= 1, f"min_variance ({min_variance}) is not in [0, 1]"

        cur_eigenvalue_sum: float = 0.0
        total_eigenvalue_sum = top_eigenvalues.sum()

        for i, eigenvalue in enumerate(top_eigenvalues):
            cur_variance = cur_eigenvalue_sum / total_eigenvalue_sum
            if cur_variance >= min_variance:
                n_components = i
                break

            cur_eigenvalue_sum += eigenvalue

        if n_components is None:
            n_components = len(top_eigenvalues)

    assert n_components is not None
    assert 0 <= n_components <= len(top_eigenvalues), n_components

    print("n_components:", n_components)
    u = top_eigenvectors[:, :n_components]  # (D, n_components)
    return u


def fda(
    train_set: MnistDataset,
    n_components: int = 2,
) -> npt.NDArray[np.float64]:
    classes: npt.NDArray[np.uint8] = np.unique(train_set.y)

    mean = train_set.x.mean(axis=0)  # (D,)
    within_class_scatter = np.zeros((len(mean), len(mean)))  # (D, D)
    between_class_scatter = np.zeros((len(mean), len(mean)))  # (D, D)
    for i in classes:
        x_cls = train_set.x_for_class(i)  # (B, D) # B is the number of samples of class i
        cls_mean = x_cls.mean(axis=0)  # (D,)

        within_class_scatter += compute_covariance(x_cls) * (len(x_cls) - 1)  # (D, D)
        between_class_scatter += len(x_cls) * np.outer(cls_mean - mean, cls_mean - mean)  # (D, D)

    m = np.linalg.pinv(within_class_scatter) @ between_class_scatter  # (D, D)

    # Compute eigenvectors
    eigenresult = np.linalg.eig(m)
    eigenvalues = eigenresult[0]  # (D,)
    eigenvectors = eigenresult[1]  # (D, D) # (D, i-th eigenvector)

    top_eigenvalue_indices = np.argsort(eigenvalues)[::-1]  # (D,) (Descending order)
    top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]  # (D, D) # (D, i-th eigenvector)
    w = top_eigenvectors[:, :n_components]  # (D, n_components)
    return w


def lda(test_set: MnistDataset, train_set: MnistDataset) -> float:
    classes: npt.NDArray[np.uint8] = np.unique(train_set.y)

    w = {}
    w0 = {}
    cov_inv = np.linalg.pinv(compute_covariance(train_set.x))  # (D, D) # To avoid recomputing
    for cls in classes:
        x_cls = train_set.x_for_class(cls)  # (B, D)
        cls_mean = x_cls.mean(axis=0)  # (D,)

        w[cls] = cov_inv @ cls_mean  # (D,)
        p = len(x_cls) / len(train_set)  # Scalar
        w0[cls] = (-0.5 * cls_mean.T @ cov_inv @ cls_mean) + np.log(p)  # Scalar

    n_correct = 0
    for x, y in zip(test_set.x, test_set.y):
        g = {}
        for cls in classes:
            g[cls] = w[cls].T @ x + w0[cls]

        y_pred = max(g, key=g.get)
        n_correct += np.allclose(y, y_pred)

    accuracy = 100 * n_correct / len(test_set)
    print(f"Accuracy: {accuracy:0.3f}%")
    return accuracy


def qda(test_set: MnistDataset, train_set: MnistDataset) -> float:
    classes: npt.NDArray[np.uint8] = np.unique(train_set.y)

    W = {}
    w = {}
    w0 = {}
    for i in classes:
        x_cls = train_set.x_for_class(i)  # (B, D)
        cls_mean = x_cls.mean(axis=0)  # (D,)

        cov_cls = compute_covariance(x_cls)  # (D, D) # To avoid recomputing
        cov_cls_inv = np.linalg.pinv(cov_cls)  # (D, D) # To avoid recomputing

        W[i] = -0.5 * cov_cls_inv  # (D, D)
        w[i] = cov_cls_inv @ cls_mean  # (D,)
        p = len(x_cls) / len(train_set)  # Scalar
        w0[i] = (
            (-0.5 * cls_mean.T @ cov_cls_inv @ cls_mean) - (0.5 * np.linalg.slogdet(cov_cls)[1]) + np.log(p)
        )  # Scalar

    n_correct = 0
    for x, y in zip(test_set.x, test_set.y):
        g = {}
        for i in classes:
            g[i] = x.T @ W[i] @ x + w[i].T @ x + w0[i]

        y_pred = max(g, key=g.get)
        n_correct += np.allclose(y, y_pred)

    accuracy = 100 * n_correct / len(test_set)
    print(f"Accuracy: {accuracy:0.3f}%")
    return accuracy


def plot_dataset(dataset: MnistDataset, title: str = ""):
    colors = [
        "steelblue",
        "indianred",
        "mediumseagreen",
        "rosybrown",
        "sandybrown",
        "darkkhaki",
        "mediumpurple",
        "cadetblue",
        "slategray",
        "teal",
        "goldenrod",
        "lightcoral",
        "darkolivegreen",
        "periwinkle",
    ]

    fig = plt.figure(layout="constrained")
    n_dims = 3 if len(dataset.x[0]) >= 3 else 2
    ax = fig.add_subplot(projection="3d" if n_dims == 3 else None)
    ax.set_title(title)

    classes = np.unique(dataset.y)
    for cls in classes:
        x = dataset.x_for_class(cls)
        color = colors[cls]
        if n_dims == 2:
            ax.scatter(x[:, 0], x[:, 1], c=color, alpha=0.5, label=cls)
        else:
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, alpha=0.5, label=cls)

    ax.legend()
    fig.savefig(f"{title}.png", dpi=300)
    plt.show()


def main():
    method = "PCA"  # PCA or FDA
    # class_counts_to_keep = None # Use this instead if you want to keep the complete dataset
    class_counts_to_keep: Union[None, Dict[int, Union[int, None]]] = {0: 100, 1: 100, 2: 100}

    x_train = load_idx("train-images.idx3-ubyte").astype(np.float64)
    y_train = load_idx("train-labels.idx1-ubyte")
    train_set = MnistDataset(x_train, y_train, class_counts_to_keep)
    train_set.preprocess()
    print("Train set:", train_set)

    x_test = load_idx("t10k-images.idx3-ubyte").astype(np.float64)
    y_test = load_idx("t10k-labels.idx1-ubyte")
    test_set = MnistDataset(x_test, y_test, class_counts_to_keep)
    test_set.preprocess()
    print("Test set:", test_set)
    
    mle(train_set)

    print()

    print(f"Using {method}")
    u = pca(train_set, n_components=17) if method == "PCA" else fda(train_set, n_components=2)

    x_train_proj = train_set.x @ u
    y_train_proj = train_set.y.copy()
    proj_train_set = MnistDataset(x_train_proj, y_train_proj)
    print("Projected train set:", proj_train_set)

    x_test_proj = test_set.x @ u
    y_test_proj = test_set.y.copy()
    proj_test_set = MnistDataset(x_test_proj, y_test_proj)
    print("Projected test set:", proj_test_set)   
    print()

    print(f"LDA on train_set after {method}")
    lda(proj_train_set, proj_train_set)
    
    print(f"LDA on test_set after {method}")
    lda(proj_test_set, proj_train_set)

    print()

    print(f"QDA on train_set after {method}")
    qda(proj_train_set, proj_train_set)

    print(f"QDA on test_set after {method}")
    qda(proj_test_set, proj_train_set)

    # Use the below command to plot the data
    plot_dataset(proj_test_set, title=method)


if __name__ == "__main__":
    main()
