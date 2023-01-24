"""
The purpose of this test set is to test all aspects of feature generation.

:author: Stippinger
"""

import numpy as np

from biometric_blender.generator_api import (
    EffectiveFeature, FeatureBlender, SegmentShuffle, make_usefulness)

from test_utilities import (
    forest_decision_feature_importance, feature_space_generator)

from scipy.stats import rankdata, uniform
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state


# SegmentShuffle

def test_get_block_sizes():
    rng = check_random_state(137)

    total = 55
    for par in [7, 4.5, uniform(7, 10)]:
        segs = SegmentShuffle(par).get_block_sizes(total, random_state=rng)
        assert np.sum(segs) == total

    par = 20
    for total in range(79, 121):
        segs = SegmentShuffle(par).get_block_sizes(total, random_state=rng)
        assert np.sum(segs) == total
        assert len(segs) == total // par


def test_transform():
    rng = check_random_state(137)

    total = 1023
    for par in [7, 4.5, uniform(7, 10)]:
        shape = np.arange(total) + 1
        segs = SegmentShuffle(par, random_state=rng).fit_transform(shape)
        assert len(np.unique(segs)) == total


# EffectiveFeature

def test_get_samples():
    n_classes = 20
    n_samples_per_class = 100

    ef = EffectiveFeature(n_classes, 0.2, random_state=137)
    ef.fit()

    samples, labels = ef.get_samples(n_samples_per_class)

    assert samples.shape == labels.shape
    assert samples.shape == (n_classes, n_samples_per_class)


# FeatureBlender

def test_transform():
    data = np.random.uniform(0, 1, size=(100, 11))

    n_features_out = 13
    fm = FeatureBlender(n_features_out=n_features_out)
    out = fm.fit_transform(data)

    assert out.shape == (100, 13)


def test_transform_iris():
    data = load_iris()

    n_features_out = 103
    fm = FeatureBlender(n_features_out=n_features_out)
    out = fm.fit_transform(data['data'], data['target'])

    expect = len(data['data']), n_features_out
    assert out.shape == expect


def test_transform_wine():
    data = load_wine()

    n_features_out = 1013
    fm = FeatureBlender(n_features_out=n_features_out)
    out = fm.fit_transform(data['data'], data['target'])

    expect = len(data['data']), n_features_out
    assert out.shape == expect


# Generator

def test_make_usefulness():
    usefulness_linear = make_usefulness('linear', 0.1, 0.9, 5)
    expect_linear = [0.9, 0.7, 0.5, 0.3, 0.1]
    assert np.allclose(usefulness_linear, expect_linear)

    usefulness_exp = make_usefulness('exponential', 0.2, 0.8, 3)
    expect_exp = [0.8, 0.4, 0.2]
    assert np.allclose(usefulness_exp, expect_exp)

    usefulness_longtailed = make_usefulness('longtailed', 0.01, 0.81, 5, 2)
    expect_longtailed = [0.81, 0.49, 0.25, 0.09, 0.01]
    assert np.allclose(usefulness_longtailed, expect_longtailed)


def test_generate_feature_space():
    for _, kw, fs in feature_space_generator():
        out_features, out_labels, out_usefulness, out_names, _, _ = fs

        n_features_out = kw['n_features_out']
        n_classes = kw['n_classes']
        n_samples_per_class = kw['n_samples_per_class']

        assert out_features.shape == \
            (n_samples_per_class * n_classes, n_features_out)
        assert out_labels.shape == \
            (n_samples_per_class * n_classes,)
        assert out_usefulness.shape == (n_features_out,)
        assert out_names.shape == (n_features_out,)


def test_usefulness_vs_importance():
    for _, _, fs in feature_space_generator():
        out_features, out_labels, out_usefulness, _, _, _ = fs

        rf = RandomForestClassifier(random_state=137)
        rf.fit(out_features, out_labels)
        fi = forest_decision_feature_importance(rf)

        out_rank = rankdata(out_usefulness)
        fi_rank = rankdata(fi)
        corr = np.corrcoef(out_rank, fi_rank)[0, 1]

        # In general, due to mixing and noise we do not expect the very
        # same shape of the out_usefulness anf fi curves but ranks should
        # match to a certain level
        assert corr > 0.25
