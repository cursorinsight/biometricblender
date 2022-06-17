"""
The purpose of this module is to experiment with the mathematical formulation
of blending

@author: Stippinger
"""

import logging
from timeit import timeit
from typing import Union

import numba as nb
import numpy as np
from numpy.random import RandomState
from tqdm import tqdm

_RANDOM_STATE_TYPE = Union[RandomState, int, None]

# Note: Weights have many 0s and there are positive and negative
#     weights too; while log(0) in X results in -np.inf, so
#     multiplication like -np.inf * 0 --> np.nan occur and
#     the matmul would eventually sum -np.inf with +np.inf --> nan.
#     We want the output to be 0 iff an element in X == 0 with
#     non-zero coefficient in weights; converting X to eps is not
#     an option if multiple weights may be applied to eps values.


strategies = ['naive_log_strategy(X, weights.T)',
              # 'matmul_int_strategy(X, weights.T)',
              'matmul_float_strategy(X, weights.T)',
              'loop_strategy(X, weights.T)',
              'numba_strategy(X, weights.T)',
              ]


def naive_log_strategy(X: np.ndarray, weights_T: np.ndarray) -> np.ndarray:
    # Fast, but incorrect if X has zero where weight. Incorrect
    #     because X == 0 to be skipped when weight_T == 0 causes nan,
    #     just like two actual X == 0 on the same row, one with positive
    #     weight and one with negative weight.
    with np.errstate(divide='ignore'):
        Xnew = np.exp(np.matmul(np.log(np.abs(X)), weights_T))
    return np.nan_to_num(Xnew, nan=0, neginf=0, posinf=0)


def matmul_int_strategy(X: np.ndarray, weights_T: np.ndarray) -> np.ndarray:
    # Quite slow. We check for matched zeros in X and weights.
    # But we screw execution time because integer matmul is inefficient.
    assert np.all(np.isfinite(X))
    with np.errstate(divide='ignore'):
        Xlog = np.nan_to_num(np.log(np.abs(X)), nan=None, neginf=0)
    Xnew = np.exp(np.matmul(Xlog, weights_T))
    Xnull = np.matmul((X == 0).astype(int), (weights_T != 0).astype(int))
    Xnew[Xnull.astype(bool)] = 0
    return Xnew


def matmul_float_strategy(X: np.ndarray, weights_T: np.ndarray) -> np.ndarray:
    # Quite fast. We check for matched zeros in X and weights
    assert np.all(np.isfinite(X))
    with np.errstate(divide='ignore'):
        Xlog = np.nan_to_num(np.log(np.abs(X)),
                             nan=np.nan, neginf=0, posinf=np.inf)
    Xnew = np.exp(np.matmul(Xlog, weights_T))
    Xnull = np.matmul((X == 0).astype(float), (weights_T != 0).astype(float))
    Xnew[Xnull.astype(bool)] = 0
    return Xnew


def loop_strategy(X: np.ndarray, weights_T: np.ndarray) -> np.ndarray:
    # Mathematically wrong, but useful for the above rules:
    #     np.power(0, 0) == 1, np.power(0., 0.) = 1.
    if X.shape[1] != weights_T.shape[0]:
        raise ValueError("Input X features does not conform with the "
                         "X used during fit.")
    XabsT = np.abs(X.T[..., np.newaxis])  # (n_features_in, n_samples)
    Xnew = np.power(XabsT[0], weights_T[0])
    for i in range(1, XabsT.shape[0]):
        Xnew *= np.power(XabsT[i], weights_T[i])
    assert Xnew.shape == (X.shape[0], weights_T.shape[1])
    return Xnew


@nb.jit(nopython=True)
def numba_accelerated_inner_loop(XabsT, weights_T, Xnew):
    for i in range(1, XabsT.shape[0]):
        Xnew *= XabsT[i] ** weights_T[i]  # power should be an ufunc


def numba_strategy(X: np.ndarray, weights_T: np.ndarray) -> np.ndarray:
    # Mathematically wrong, but useful for the above rules:
    #     np.power(0, 0) == 1, np.power(0., 0.) = 1.
    if X.shape[1] != weights_T.shape[0]:
        raise ValueError("Input X features does not conform with the "
                         "X used during fit.")
    XabsT = np.abs(X.T[..., np.newaxis])  # (n_features_in, n_samples, 1)
    Xnew = np.power(XabsT[0], weights_T[0])
    assert Xnew.shape == (X.shape[0], weights_T.shape[1])
    numba_accelerated_inner_loop(XabsT, weights_T, Xnew)
    assert Xnew.shape == (X.shape[0], weights_T.shape[1])
    return Xnew


def init_data(large=True, seed=137):
    rs = np.random.RandomState(seed)
    if large:
        X = rs.normal(size=(1600, 1001))
        weights = np.zeros((13000, 1001))
    else:
        X = rs.normal(size=(160, 101))
        weights = np.zeros((300, 101))
    X.flat[rs.choice(X.size, size=X.size // 4, replace=False)] = 0
    weights.flat[rs.choice(weights.size, size=X.size // 100, replace=False
                           )] = rs.normal(size=X.size // 100)
    return X, weights


def time_strategies(large=True):
    """Measure execution time of different strategies"""
    X, weights = init_data(large=False)
    numba_strategy(X, weights.T)  # do compilation before timing
    X, weights = init_data(large=large)
    env = globals().copy()
    env.update({'X': X, 'weights': weights})
    timing = {}
    for q in tqdm(strategies):
        t = timeit(q, number=1, globals=env)
        logging.log(logging.INFO, msg='call: {}, time: {}'.format(q, t))
        timing[q] = t
    return timing


def test_strategies(large=False):
    """Compare output of different strategies"""
    X, weights = init_data(large=large)
    env = globals().copy()
    env.update({'X': X, 'weights': weights})
    result = {}
    for q in tqdm(strategies):
        result[q] = eval(q)
    r0 = result['numba_strategy(X, weights.T)']
    for q, r in result.items():
        if q.startswith('naive'):
            continue
        assert np.allclose(r, r0)  # won't pass for naive
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    timing = time_strategies(large=False)
    print(timing)
