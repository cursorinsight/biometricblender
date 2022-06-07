import numpy as np

from biometric_blender.generator_api import generate_feature_space
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

def forest_decision_feature_importance(
    rf: RandomForestClassifier
) -> np.ndarray:
    """
    Extract occurrence-count in decision based feature importances
    from a fitted forest
    """
    n_features = rf.n_features_in_

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


def feature_space_generator(n_labels=100,
                            n_samples_per_label=16,
                            n_true_features=9,
                            n_features_out=1013):

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              min_usefulness=1,
              max_usefulness=1,
              n_fake_features=0)
    fs = generate_feature_space(**kw)
    yield 'all useful', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              min_usefulness=0.1,
              max_usefulness=1,
              n_fake_features=0)
    fs = generate_feature_space(**kw)
    yield 'basic', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              polynomial=True)
    fs = generate_feature_space(**kw)
    yield 'polynomial', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              relative_usefulness_content=uniform(0.6, 0.4),
              blending_mode='logarithmic')
    fs = generate_feature_space(**kw)
    yield 'logarithmic', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              blending_mode='logarithmic',
              relative_usefulness_content=randint(1, 2))
    fs = generate_feature_space(**kw)
    yield 'noiseless logarithmic', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              n_fake_features=0,
              relative_usefulness_content=uniform(0.6, 0.4),
              blending_mode='logarithmic')
    fs = generate_feature_space(**kw)
    yield 'true logarithmic', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              n_fake_features=0,
              blending_mode='logarithmic',
              relative_usefulness_content=randint(1, 2))
    fs = generate_feature_space(**kw)
    yield 'pure logarithmic', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              location_ordering_extent=3)
    fs = generate_feature_space(**kw)
    yield 'ordered', kw, fs

    kw = dict(n_labels=n_labels,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_features_out=n_features_out,
              location_sharing_extent=3)
    fs = generate_feature_space(**kw)
    yield 'shared', kw, fs
