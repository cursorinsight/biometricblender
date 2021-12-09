"""
The purpose of this module is to evaluate a feature set using a random forest

@author: Stippinger
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def forest_decision_feature_importance(
        rf: RandomForestClassifier
        ) -> np.ndarray:
    """
    Extract occurrence-count in decision based feature importances
    from a fitted forest
    """
    n_features = rf.n_features_
    counts = np.zeros(n_features, dtype=int)
    for tree in rf.estimators_:
        tree = tree  # type: DecisionTreeClassifier
        # actual type of tree: rf.base_estimator_
        children_left = tree.tree_.children_left  # type: np.ndarray
        children_right = tree.tree_.children_right  # type: np.ndarray
        is_split_node = children_left != children_right
        feature = tree.tree_.feature  # type: np.ndarray
        np.add.at(counts, feature[is_split_node], 1)
    return counts
