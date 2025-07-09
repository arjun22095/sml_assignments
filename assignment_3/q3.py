from itertools import chain, combinations
from typing import Callable, Optional

import numpy as np
import pandas as pd


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def compute_gini_impurity(labels: pd.Series) -> float:
    assert len(labels) > 0
    probs = labels.value_counts(normalize=True)
    gini_impurity = sum([probs[i] * (1 - probs[i]) for i in range(len(probs))])
    # return 1.0 - sum(probs**2)
    return gini_impurity


def compute_split_condition_for_numerics(data: pd.Series, col_name: str):
    best_impurity = float("inf")
    best_split_condition = None

    unique_vals = sorted(data.unique().tolist())
    for i in range(len(unique_vals) - 1):
        mid = (unique_vals[i] + unique_vals[i + 1]) / 2

        left_mask = data < mid
        left_data = data[left_mask]
        right_data = data[~left_mask]

        impurity_left = compute_gini_impurity(left_data["label"])
        impurity_right = compute_gini_impurity(right_data["label"])
        total_impurity = (impurity_left * len(left_data) + impurity_right * len(right_data)) / len(data)

        if total_impurity < best_impurity:
            best_impurity = total_impurity

            def best_split_condition(x: pd.Series) -> bool:
                return x[col_name] < mid

    return best_split_condition, best_impurity


def compute_split_condition_for_categorical(data: pd.Series, col_name: str):
    best_impurity = float("inf")
    best_split_condition = None

    unique_vals = sorted(data.unique().tolist())
    for _subset in powerset(unique_vals):
        subset = frozenset(_subset)

        left_mask = data.isin(subset)
        left_data = data[left_mask]
        right_data = data[~left_mask]

        impurity_left = compute_gini_impurity(left_data["label"])
        impurity_right = compute_gini_impurity(right_data["label"])
        total_impurity = (impurity_left * len(left_data) + impurity_right * len(right_data)) / len(data)

        if total_impurity < best_impurity:
            best_impurity = total_impurity

            def best_split_condition(x: pd.Series) -> bool:
                return x[col_name] in subset

    return best_split_condition, best_impurity


def compute_split_condition(dataset: pd.DataFrame):
    best_impurity = float("inf")
    best_split_condition = None

    for col, data in dataset.items():
        # print(f"{type(data)=}")
        # print(f"||||| iterator, {col=}, {data=} |||||\n\n")
        print(col, data)
        if np.issubdtype(data.dtype, np.number):
            split_condition, impurity = compute_split_condition_for_numerics(data, col)
        else:
            split_condition, impurity = compute_split_condition_for_categorical(data, col)

        if impurity < best_impurity:
            best_impurity = impurity
            best_split_condition = split_condition

    return best_split_condition, impurity


class TreeNode:
    def __init__(self):
        self.condition: Callable[[pd.Series], bool] = ...

    def apply(
        self,
        x: pd.Series,
    ):
        return self.condition(x)

    __call__ = apply

    def __str__(self) -> str:
        return "TreeNode"


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_in_leaf: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_in_leaf = min_samples_in_leaf

        self._root = TreeNode()


def load_dataset(path: str):
    dataset = pd.read_csv(path)
    dataset.rename(columns={"Buy Computer": "label"}, inplace=True)
    return dataset


def main():
    dataset = load_dataset("dataset.csv")
    print(dataset)
    x = compute_split_condition(dataset)
    print(x)
    print("computed")
    classifier = DecisionTreeClassifier()


if __name__ == "__main__":
    main()
