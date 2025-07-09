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
    return sum([p * (1 - p) for p in probs])


def compute_split_condition_for_numerics(dataset: pd.DataFrame, col_name: str, min_samples_in_leaf=1):
    # print(f"Computing split condition for {col_name}")
    best_impurity = float("inf")
    best_split_mid = 0

    unique_vals = sorted(dataset[col_name].unique().tolist())
    for i in range(len(unique_vals) - 1):
        mid = (unique_vals[i] + unique_vals[i + 1]) / 2

        left_mask = dataset[col_name] < mid
        left_data = dataset[left_mask]
        right_data = dataset[~left_mask]

        impurity_left = compute_gini_impurity(left_data["label"])
        impurity_right = compute_gini_impurity(right_data["label"])
        total_impurity = (impurity_left * len(left_data) + impurity_right * len(right_data)) / len(dataset)

        if total_impurity < best_impurity:
            if len(left_data) < min_samples_in_leaf or len(right_data) < min_samples_in_leaf:
                continue

            best_impurity = total_impurity
            best_split_mid = mid

    if best_impurity == float("inf"):
        # print("Returning none")
        return None, None

    # Returns a mask for the left data
    def best_split_condition(df: pd.DataFrame) -> pd.Series:
        return df[col_name] < best_split_mid

    return best_split_condition, best_impurity


def compute_split_condition_for_categorical(dataset: pd.DataFrame, col_name: str, min_samples_in_leaf=1):
    # print(f"Computing split condition for {col_name}")
    best_impurity = float("inf")
    best_subset = None
    unique_vals = sorted(dataset[col_name].unique().tolist())

    for _subset in powerset(unique_vals):
        subset = frozenset(_subset)
        if len(subset) == len(unique_vals):
            continue

        left_mask = dataset[col_name].isin(subset)
        left_data = dataset[left_mask]
        right_data = dataset[~left_mask]

        impurity_left = compute_gini_impurity(left_data["label"])
        impurity_right = compute_gini_impurity(right_data["label"])
        total_impurity = (impurity_left * len(left_data) + impurity_right * len(right_data)) / len(dataset)
        if total_impurity < best_impurity:
            if len(left_data) < min_samples_in_leaf or len(right_data) < min_samples_in_leaf:
                continue

            best_impurity = total_impurity
            best_subset = subset

    if best_subset is None:
        return None, None

    # Returns a mask for the left data
    def best_split_condition(df: pd.DataFrame) -> pd.Series:
        return df[col_name] in (best_subset)

    return best_split_condition, best_impurity


def compute_split_condition(dataset: pd.DataFrame, min_samples_in_leaf=None, num_features_for_rf : int = -1):
    best_impurity = compute_gini_impurity(dataset)  # Impurity without splitting
    best_split_condition = None  # Initialize it to -> do not split!
    best_feature = None
    if best_impurity == 0:
        # print("Already a pure dataset!")
        return best_split_condition, best_impurity

    columns = [col for col in dataset.columns if col != "label"]
    if (num_features_for_rf != -1):
        columns = np.random.choice(columns, size=num_features_for_rf, replace=False)
        print("Since its a Random Forest, I am only considering the following features for this split")
        print(columns)

    for col in columns:
        data = dataset[col]

        if np.issubdtype(data.dtype, np.number):
            split_condition, impurity = compute_split_condition_for_numerics(dataset, col, min_samples_in_leaf)
        else:
            split_condition, impurity = compute_split_condition_for_categorical(dataset, col, min_samples_in_leaf)

        if impurity is not None and impurity < best_impurity:
            best_impurity = impurity
            best_split_condition = split_condition
            best_feature = col

    return best_split_condition, best_impurity, best_feature


class TreeNode:
    def __init__(self):
        self.condition: Callable[[pd.Series], bool] = ...
        self.left = None
        self.right = None
        self.parent = None
        self.split_feature = None  # Which feature are you splitting the Tree on
        self.impurity = None
        self.predicted_label = None
        self.tag = None
        self.depth = 0

    def is_leaf(self) -> bool:
        return (not self.right) and (not self.left)

    def delete(self):
        if self.parent is not None:
            if self.parent.left == self:
                self.parent.left = None
            else:
                self.parent.right = None

    def init_node(self, df: pd.DataFrame, min_samples_in_leaf=None, num_features_for_rf : int = -1):
        print(f"{num_features_for_rf}=")
        self.condition, self.impurity, self.split_feature = compute_split_condition(df, min_samples_in_leaf, num_features_for_rf)

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        if self.condition is None:
            raise ValueError("Condition is not set in this node.")

        mask = df.apply(self.condition, axis=1)  # Generate boolean mask
        return mask

    def get_left_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.condition is None:
            raise ValueError("Condition is not set in this node.")

        mask = self.get_mask(df)
        return df[mask]

    def get_right_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.condition is None:
            raise ValueError("Condition is not set in this node.")

        mask = self.get_mask(df)
        return df[~mask]

    def __str__(self) -> str:
        return "TreeNode"


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_in_leaf: Optional[int] = 1,
        num_features_for_rf : int = -1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_in_leaf = min_samples_in_leaf
        self.num_leaves = 0
        self.num_features_for_rf = num_features_for_rf
        self.root = TreeNode()

    def evaluate_tree(self, node: TreeNode, df: pd.DataFrame, predicted_labels: dict = -1, display=False):
        if node is None:
            return

        if node.is_leaf():
            # assert node.split_feature is None
            if display:
                print(f"\n\nLeaf ({node.tag}) | Depth = {node.depth} | Predicts = {node.predicted_label}")
                print(df)

            if predicted_labels != -1:
                for i in df.index:
                    predicted_labels[i] = node.predicted_label

            return

        if display:
            print("Node split feature is ", node.split_feature)

        self.evaluate_tree(node.left, node.get_left_dataset(df), predicted_labels, display)
        self.evaluate_tree(node.right, node.get_right_dataset(df), predicted_labels, display)

    def predict_labels(self, df: pd.DataFrame, display=False) -> dict:
        predicted_labels = dict()
        self.evaluate_tree(self.root, df, predicted_labels, display)
        return predicted_labels

    def set_leaf_labels(self, node: TreeNode, df: pd.DataFrame):
        if node is None:
            return

        if node.is_leaf():
            self.num_leaves += 1
            node.tag = self.num_leaves
            node.predicted_label = df["label"].mode()[0]
            return

        self.set_leaf_labels(node.left, node.get_left_dataset(df))
        self.set_leaf_labels(node.right, node.get_right_dataset(df))

    def helper_create(self, node: TreeNode, df: pd.DataFrame):
        if (
            (df is None or df.empty)
            or (compute_gini_impurity(df["label"]) == 0)
            or (len(df) <= self.min_samples_in_leaf)
            or (self.max_depth and node.depth == self.max_depth)
        ):
            return

        node.init_node(df, self.min_samples_in_leaf, self.num_features_for_rf)
        if node.split_feature is None:
            # DO NOT DELETE NODE!
            # Because each leaf node must have a tag, depth and a label
            # but each leaf node SHOULD NOT HAVE a split feature
            return

        left_dataset = node.get_left_dataset(df)
        node.left = TreeNode()
        node.left.depth = node.depth + 1
        node.left.parent = node
        self.helper_create(node.left, left_dataset)

        right_dataset = node.get_right_dataset(df)
        node.right = TreeNode()
        node.right.depth = node.depth + 1
        node.right.parent = node
        self.helper_create(node.right, right_dataset)

    def create_decision_tree(self, dataset: pd.DataFrame):
        self.helper_create(self.root, dataset)
        self.set_leaf_labels(self.root, dataset)


class RandomForest:
    def __init__(
        self,
        num_classifiers: int,
        training_dataset: pd.DataFrame,
        num_features_for_rf : int = -1,
        max_depth: Optional[int] = None,
        min_samples_in_leaf: Optional[int] = 1,
    ) -> None:
        self.num_classifiers = num_classifiers
        self.training_dataset = training_dataset
        self.max_depth = max_depth
        self.min_samples_in_leaf = min_samples_in_leaf
        self.classifiers: list[DecisionTreeClassifier] = []
        self.num_features_for_rf = num_features_for_rf
        self.bootstrapped_rows = []  # will be a list of sets

        for i in range(self.num_classifiers):
            rows_to_keep_for_bootstrapping = np.random.choice(
                training_dataset.index, size=len(training_dataset), replace=True,
            )
            self.bootstrapped_rows.append(set(rows_to_keep_for_bootstrapping))

            # bootstrapped_df = self.training_dataset.sample(n=len(self.training_dataset), replace=True) (Inefficient)
            bootstrapped_df = training_dataset.loc[rows_to_keep_for_bootstrapping]
            print(f"\nBoostrapped Dataset {i} is")
            print(bootstrapped_df)
            classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_in_leaf=min_samples_in_leaf, num_features_for_rf=self.num_features_for_rf)
            classifier.create_decision_tree(bootstrapped_df)
            self.classifiers.append(classifier)

    def calculate_oob(self):
        all_rows = set(self.training_dataset.index)
        predictions = dict()
        for i in range(self.num_classifiers):
            bagged_rows_for_c = self.bootstrapped_rows[i]
            out_of_bag_rows_for_c = all_rows - bagged_rows_for_c
            for row_index in out_of_bag_rows_for_c:
                if row_index not in predictions:
                    predictions[row_index] = []

                row = self.training_dataset.loc[[row_index]]
                predictions[row_index].append(
                    self.classifiers[i].predict_labels(row),
                )

        misclassified = 0
        total = len(predictions)
        for row_idx in predictions:
            majority_predicted_label = pd.Series(predictions[row_idx]).mode()
            actual_label = self.training_dataset.loc[row_idx, "label"]
            misclassified += (majority_predicted_label != actual_label)

        oob = misclassified / total
        print("OOB Error is ", oob)
        return oob

    def predict_label(self, test_df: pd.DataFrame):
        print("\nTraining Dataset (One to predict) is")
        # print(test_df)

        predictions = [tree.predict_labels(test_df) for tree in self.classifiers]
        predictions_df = pd.DataFrame(predictions).T
        print("\nPredictions of all Classifiers were ")
        predictions_df = predictions_df.loc[test_df.index]  # Sorting by the indices of the test dataframe
        # print(predictions_df)
        print("\n\nTaking the majority of each for our final predictions")
        combined = predictions_df.mode(axis=1)[0]
        combined = combined.loc[test_df.index]  # Sorting by the indices of the test dataframe
        # print(combined)


def load_dataset(path: str):
    dataset = pd.read_csv(path)
    if "Buy Computer" in dataset.columns:
        dataset.rename(columns={"Buy Computer": "label"}, inplace=True)
    return dataset


def main():
    pd.set_option("display.max_rows", None)

    train_df = load_dataset("training_dataset.csv")
    rf = RandomForest(num_classifiers=10, num_features_for_rf=2, training_dataset=train_df)
    rf.predict_label(train_df)
    rf.calculate_oob()

if __name__ == "__main__":
    main()
