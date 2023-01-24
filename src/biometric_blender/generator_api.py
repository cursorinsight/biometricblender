"""
The purpose of this module is to generate a plausible random feature set

:author: Stippinger
"""

import warnings
from typing import Union, Tuple, Iterable, Sequence

import numpy as np
from numpy.random import RandomState
from scipy import sparse, stats
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted

_RANDOM_STATE_TYPE = Union[RandomState, int, None]
_MSG_INVALID_BLENDING = ("Invalid blending mode, choose from "
                         "'linear' and 'logarithmic'.")


class SegmentShuffle(BaseEstimator, TransformerMixin):
    """
    Transformer class that allows shuffling blocks
    """

    def __init__(
            self,
            length_dist: Union[
                int, float, stats.rv_continuous, stats.rv_discrete],
            longer: str = 'distribute',
            random_state: Union[np.random.RandomState, None] = None):
        """
        :param length_dist:
          distribution for segment size
        :param longer: where to place longer block:
          {'end', 'start', 'random', 'distribute'}, optional,
          default: 'distribute'
        :param random_state: seed or RandomState instance, use None to
          auto-seed
        """
        self.length_dist = length_dist
        self.longer = longer
        self.random_state = random_state

    def fit(self, X, y=None):
        del X, y
        return self

    def get_block_sizes(self, total_length, longer='distribute',
                        random_state=None) -> np.ndarray:
        """
        Get random block sizes.

        Notes:
          One element may be out of the given distribution
          (because the remainder got added to it).

        :param int total_length: data length
        :param str longer: where to place longer block
          {'end', 'start', 'random', 'distribute'}, optional,
          default: 'distribute'
        :param np.random.RandomState random_state:
        :return np.ndarray[int]: block sizes
        """
        length_dist = self.length_dist
        if random_state is None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = check_random_state(random_state)
        try:
            if hasattr(length_dist, 'mean') and hasattr(length_dist, 'rvs'):
                # Take twice as many samples as thought sufficient
                with np.errstate(divide='ignore'):
                    # querying randint's higher order stats may issue a warning
                    mean = length_dist.mean()
                n = int(2 * total_length / mean)
                part = length_dist.rvs(size=n, random_state=random_state
                                       ).astype(int)
                part = part[0 < part]
                n = len(part)
            else:
                # Take uniform blocks
                n = int(total_length / length_dist) + 1
                part = np.full(n, length_dist)
        except TypeError as e:
            msg = ('Length_dist must either be a number or be compatible with'
                   'the scipy.stats distribution API, got instance of %s'
                   % type(length_dist))
            raise TypeError(msg) from e
        # Find the sufficient samples
        cum = np.cumsum(part).astype(int)
        trunc = np.searchsorted(cum, total_length, side='right')
        # Note: cum[trunc-1] <= total_length < cum[trunc], 0<=trunc<=n
        part = np.ediff1d(cum, to_begin=[int(part[0])])[:trunc]
        # If no samples approximate with singleton
        if trunc == 0:
            return np.array([total_length])
        # Set size of many longer blocks
        if longer == 'distribute':
            diff = total_length - cum[trunc - 1]
            base = diff // trunc
            n_extra = int(diff - base * trunc)
            part += base
            part[:n_extra] += 1
            part = random_state.permutation(part)
        # or set the size of single longer block
        else:
            if trunc == 0:
                trunc = 1
            part[-1] = total_length - cum[trunc - 1]

            if longer == 'start':
                idx = 0
            elif longer == 'end':
                idx = -1
            elif longer == 'random':
                idx = random_state.randint(0, len(part))
            else:
                raise ValueError

            part[-1], part[idx] = part[idx], part[-1]

        return part

    @staticmethod
    def get_block_bounds(block_sizes: Sequence[int]) -> np.ndarray:
        n_blocks = len(block_sizes)
        bounds = np.zeros((n_blocks, 2), dtype=int)
        bounds[:, 1] = np.cumsum(block_sizes)
        bounds[1:, 0] = bounds[:-1, 1]
        return bounds

    def transform(self, X):
        longer = self.longer
        random_state = check_random_state(self.random_state)

        blocks0 = self.get_block_sizes(len(X), longer=longer,
                                       random_state=random_state)
        bounds0 = self.get_block_bounds(blocks0)

        # Note: instead of accumulating sizes of random blocks,
        #       we chose to store the mapping in memory
        n_blocks = len(blocks0)
        mapping = random_state.permutation(n_blocks)
        blocks1 = blocks0[mapping]  # blocks1[i] == blocks0[mapping[i]]
        bounds1 = self.get_block_bounds(blocks1)

        result = np.copy(X)
        for i in range(n_blocks):
            s0, e0 = bounds0[mapping[i]]
            s1, e1 = bounds1[i]
            result[s0:e0] = X[s1:e1]

        return result


class EffectiveFeature(object):
    """
    Class to simulate a standalone hidden feature that is effective
    in class (or label) identification

    :ivar np.ndarray[float] locations_:
      the characteristic trait of classes, i.e., location of the feature value
      for specific labels, shape (n_classes, )
    :ivar np.ndarray[float] scales_:
      the amplitude of the sampling noise (uncertainty of reproduction),
      i.e., scale of different samples for the same class (or label),
      shape (n_classes, )
    :ivar float range_:
      the approximate scale of the feature in the population, i.e., the scale
      of the locations plus the scale of the sampling noise
    """

    def __init__(
            self,
            n_classes: int,
            extent: float,
            location_distribution: stats.rv_continuous = stats.norm,
            scale_distribution: stats.rv_continuous = stats.uniform(0.5, 1.0),
            sampling_distribution: stats.rv_continuous = stats.norm,
            location_ordering_extent: int = 0,
            location_sharing_extent: int = 0,
            random_state: _RANDOM_STATE_TYPE = None):
        """
        Class to simulate one feature that is effective in class (or label)
        identification

        Notes:
          One can use higher values for `location_ordering_extent` and
          `location_sharing_extent` to implicitly increase correlation between
          standalone hidden features.

        Examples:
          * location_distribution:
            * constrained: uniform, beta,
            * unconstrained: normal, pareto (heavy-tailed)
          * sampling_distribution:
            * constrained: uniform
            * unconstrained: normal, cauchy
          * scale_distribution:
            * constrained: uniform, beta, >0
            * unconstrained: exponential, pareto (heavy-tailed) > 0

        :param n_classes: number of classes (or labels) of the classification
          problem
        :param extent: average extent of class traits in the feature,
          interpreted as a multiple of standard deviation of the location
          distribution, advised values lie between 0.01 and 10
        :param location_distribution: distribution type of the characteristic
          trait of classes, i.e., location of the feature value for specific
          labels (`location` parameter to `sampling_distribution˙)
        :param scale_distribution: frozen pre-parametrized distribution of
          the amplitude of the sampling noise (uncertainty of reproduction),
          i.e., scale of different samples for the same class (`scale`
          parameter to `sampling_distribution˙)
        :param sampling_distribution: distribution type of the uncertainty of
          reproduction, i.e., the noise for different samples from the same
          class (or label)
        :param location_ordering_extent: average number of consecutive
          locations, use 0 for no explicit ordering, use -1 for all
          locations to be ordered increasingly, use any other negative
          value to define a fixed number (absolute value, discouraged) of
          consecutive locations
        :param location_sharing_extent: average number of classes sharing a
          common location, use 0 for none, use any negative value to define
          a fixed number (absolute value, discouraged) of classes sharing a
          common location
        :param random_state: seed or RandomState instance, use None to
          auto-seed
        """
        self.n_classes = n_classes
        self.extent = extent
        self.random_state = random_state
        self.location_distribution = location_distribution
        self.scale_distribution = scale_distribution
        self.sampling_distribution = sampling_distribution
        self.location_ordering_extent = location_ordering_extent
        self.location_sharing_extent = location_sharing_extent

    __doc__ = __doc__ + __init__.__doc__

    def fit(self, random_state: RandomState = None):
        """
        Set the parameters for sample generation

        :param random_state: seed or RandomState instance, use None to
          instance-default
        :return: self: EffectiveFeature
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)
        mean_of_sampling_std = (self.sampling_distribution.std()
                                * self.scale_distribution.mean())
        location_std = self.location_distribution.std()
        scaling_for_sampling = (self.extent * location_std
                                / mean_of_sampling_std)
        if self.location_sharing_extent:
            if 0 < self.location_sharing_extent:
                # mean number
                loc_share = stats.randint(1, 2 * self.location_sharing_extent)
            else:
                # exact number (absolute value), discouraged
                loc_share = -self.location_sharing_extent
            seg = SegmentShuffle(loc_share, random_state=random_state)
            blocks = seg.get_block_sizes(self.n_classes)
            n_unique_locations = len(blocks)
            assignment = np.zeros(self.n_classes, dtype=int)
            for i, (s, e) in enumerate(seg.get_block_bounds(blocks)):
                # keep block id ordered, it may be needed for loc. ordering
                assignment[s:e] = i
        else:
            n_unique_locations = self.n_classes
            assignment = slice(None)
        locations = self.location_distribution.rvs(
            size=n_unique_locations, random_state=random_state)[assignment]
        # Note: we could apply ordering based on block id (`assignment[]` <- i)
        if self.location_ordering_extent == 0:
            self.locations_ = random_state.permutation(locations)  # unordered
        elif self.location_ordering_extent == -1:
            self.locations_ = np.sort(locations)  # fully ordered
        else:
            if 0 < self.location_ordering_extent:
                # mean number
                loc_order = stats.randint(1, 2 * self.location_ordering_extent)
            else:
                # exact number (absolute value), discouraged
                loc_order = -self.location_ordering_extent
            self.locations_ = (
                SegmentShuffle(loc_order, random_state=random_state
                               ).fit_transform(np.sort(locations)))
        self.scales_ = scaling_for_sampling * self.scale_distribution.rvs(
            size=self.n_classes, random_state=random_state)
        self.range_ = np.linalg.norm(
            [location_std, scaling_for_sampling * mean_of_sampling_std], ord=2)

    def get_samples(self, n_samples_per_class: int,
                    random_state: RandomState = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples

        Notes:
          You might want to flatten the output arrays

        :param n_samples_per_class: number of samples per class
        :param random_state:
        :return:
          * X: np.ndarray
            samples, shape (n_classes, n_samples_per_class)
          * y: np.ndarray
            labels, shape (n_classes, n_samples_per_class)
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        check_is_fitted(self)
        # Note: we transposed parameters below to get samples from same class
        #       next to each other (check tests too), this is less suited for
        #       simple K-fold (without shuffling and stratification)
        X = self.sampling_distribution.rvs(
            loc=self.locations_[:, np.newaxis],
            scale=self.scales_[:, np.newaxis],
            size=(self.n_classes, n_samples_per_class),
            random_state=random_state)
        y = np.repeat(np.arange(self.n_classes)[:, np.newaxis],
                      n_samples_per_class, axis=1)
        return X, y


class FeatureBlender(BaseEstimator, TransformerMixin):
    """
    Class to transform input features into a random combination of them

    Notes:
      Consider normalizing the feature values before blending.

    :ivar int n_features_in_:
        number of input features
    :ivar np.ndarray[float] weights_:
        weight of input features to produce output features,
        shape (n_features_out, n_features_in_)
    """

    def __init__(
            self,
            count_distribution: stats.rv_discrete = stats.randint(5, 11),
            alpha: int = 1,
            n_features_out: int = 10,
            blending_mode: str = 'linear',
            sparse_weights: bool = True,
            random_state: _RANDOM_STATE_TYPE = None
    ):
        """
        Class to transform input features into a random combination of them

        :param count_distribution: frozen pre-parametrized distribution of the
          number input of features taking part in one specific output feature
        :param alpha: parameter for the Dirichlet distribution that determines
          weights for input features in one specific output feature
        :param n_features_out: number of output features to produce
        :param blending_mode: "linear" makes features using linear combination
          (cf. central limit theorem), "logarithmic" makes features by
          multiplication (resulting distribution approximates log-normal)
        :param sparse_weights: whether to represent weights as a sparse matrix
        :param random_state: seed or RandomState instance, use None to
          auto-seed
        """
        self.count_distribution = count_distribution
        self.alpha = alpha
        self.weight_distribution = stats.dirichlet
        self.n_features_out = n_features_out
        self.blending_mode = blending_mode
        self.sparse_weights = sparse_weights
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Set up the transformation of input features

        :param X: input features to be combined, shape (n_samples, n_features)
        :param y: not used
        :return: self: FeatureBlender
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)
        if self.blending_mode not in ['linear', 'logarithmic']:
            raise ValueError(_MSG_INVALID_BLENDING)
        n_features_in = X.shape[-1]
        counts = self.count_distribution.rvs(size=self.n_features_out,
                                             random_state=random_state)
        counts = np.minimum(counts, n_features_in)

        def placements() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
            alpha = self.alpha
            weight_distribution = self.weight_distribution
            for i_row, c in enumerate(counts):
                i_col = random_state.choice(n_features_in, c, replace=False)
                distribution = weight_distribution.rvs(
                    np.broadcast_to(alpha, c),
                    size=(),
                    random_state=random_state)
                signs = random_state.randint(0, 2, size=c) * 2 - 1
                yield i_col, distribution * signs

        if self.sparse_weights:
            col, data = map(np.concatenate, zip(*placements()))
            iptr = np.concatenate(([0], np.cumsum(counts)))
            weights = sparse.csr_matrix(
                (data, col, iptr), shape=(self.n_features_out, n_features_in))
        else:
            weights = np.zeros((self.n_features_out, n_features_in))
            for i_row, (i_col, val) in enumerate(placements()):
                weights[i_row, i_col] = val
        self.n_features_in_ = n_features_in
        self.weights_ = weights
        return self

    def transform(self, X, y=None, amplitude_like: bool = False) -> np.ndarray:
        """
        Do the transformation of input features

        :param X: input features to be combined,
          shape (n_samples, n_features_in)
        :param y: not used
        :param amplitude_like: transform quantities where amplitudes are
          summed, i.e., all weights considered positive
        :return: Xnew: np.ndarray
          combined features, shape (n_samples, n_features_out)
        """
        # shape (n_samples, n_features_in)
        X = check_array(X)  # type: np.ndarray
        assert X.shape[1] == self.n_features_in_, "number of input features"
        if amplitude_like:
            # shape (n_features_in, n_features_out)
            weights_T = np.abs(self.weights_.T)
        else:
            weights_T = self.weights_.T
        assert -1 <= np.min(weights_T) and np.max(weights_T) <= 1, 'weights'
        if self.blending_mode == 'linear':
            Xnew = X @ weights_T
        elif self.blending_mode == 'logarithmic':
            # Note: Weights have many 0s and there are positive and negative
            #     weights too; while log(0) in X results in -np.inf, so
            #     multiplications like -np.inf * 0 --> np.nan occur and
            #     the matmul would eventually sum -np.inf with +np.inf --> nan.
            #     We want the output to be 0 iff. an element in X == 0 with
            #     non-zero coefficient in weights; converting X to eps is not
            #     an option when multiple weights may be applied to eps values.
            # Insufficient for general np.ndarray:
            #     Xnew = np.exp(np.log(np.abs(X)) @ weights_T)
            # Mathematically wrong, but useful for the above rules:
            #     np.power(0, 0) == 1, np.power(0., 0.) == 1.,
            #     multiplying anything (inf, nan) with sparse zero == 0
            #     therefore we force casting to sparse.
            # We can construct a mathematically exact alternative that zeroes
            #     out the corresponding elements, but that requires two matmul:
            #     one for the values and one for the masking.
            with np.errstate(divide='ignore'):
                Xnew = np.exp(np.log(np.abs(X)) @
                              sparse.csc_matrix(np.abs(weights_T)))
        else:
            raise ValueError(_MSG_INVALID_BLENDING)
        return Xnew


class NoiseBlender(BaseEstimator, TransformerMixin):
    """
    Class to transform features into a their noisy realization

    Notes:
      Please make sure that the amplitude of the noise distribution is matched
      to the feature values.

    :ivar int n_features_in_:
      number of input features
    :ivar np.ndarray[float] weights_:
      0 <= weight <= 1 of input features in output features,
      shape (n_features_in_)
    """

    def __init__(
            self,
            noise_distribution: stats.rv_continuous = stats.norm,
            blending_mode: str = 'linear',
            random_state: _RANDOM_STATE_TYPE = None
    ):
        """
        Class to transform features into a their noisy realization

        :param noise_distribution: frozen pre-parametrized distribution of the
          number input of features taking part in one specific output feature
        :param blending_mode: 'linear' makes features using linear combination
          (cf. central limit theorem), 'logarithmic' makes features by
          multiplication (resulting distribution approximates log-normal)
        :param random_state: seed or RandomState instance, use None to
          auto-seed
        """
        self.noise_distribution = noise_distribution
        self.blending_mode = blending_mode
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Set up the transformation of input features

        :param X: input features to be combined, shape (..., n_features)
        :param y: weights of data
        :return: self: FeatureBlender
        """
        check_random_state(self.random_state)
        X = check_array(X)
        if self.blending_mode not in ['linear', 'logarithmic']:
            raise ValueError(_MSG_INVALID_BLENDING)
        self.n_features_in_ = X.shape[-1]
        weights = np.broadcast_to(y, (self.n_features_in_,))
        if np.any(weights < 0) or np.any(1 < weights):
            raise ValueError("Variable y must contain valid weights 0<=y<=1")
        self.weights_ = weights
        return self

    def transform(self, X, y=None, amplitude_like: bool = False) -> np.ndarray:
        """
        Do the transformation of input features

        :param X: input features to be combined, shape (..., n_features)
        :param y: not used
        :param amplitude_like: transform quantities where amplitudes are
          summed, i.e., all weights considered positive
        :return: Xnew: np.ndarray
          combined features, shape (..., n_features_out)
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X)  # type: np.ndarray
        assert X.shape[1] == self.n_features_in_, "number of input features"
        weights = self.weights_
        assert np.all(0 <= weights) and np.all(weights <= 1), 'weights'
        if amplitude_like:
            noise = 0
        else:
            noise = self.noise_distribution.rvs(size=X.shape,
                                                random_state=random_state)

        if self.blending_mode == 'linear':
            Xnew = X * weights + noise * (1 - weights)
        elif self.blending_mode == 'logarithmic':
            Xnew = (np.where(X == 0, 1., np.power(np.abs(X), weights))
                    * np.where(noise == 0,
                               1.,
                               np.power(np.abs(noise), 1 - weights)))
        else:
            raise ValueError(_MSG_INVALID_BLENDING)
        return Xnew


def poly_features(X: np.ndarray, degree: int):
    """
    Provide all possible polynomial terms of exactly the given `degree`.
    Terms are normalized by taking root `degree`.

    :param X: input, shape (n_samples, n_features_in)
    :param degree: total degree of polynomials
    :return: output, shape (n_samples, n_features_out)
    """
    pf = PolynomialFeatures(degree=degree, interaction_only=False,
                            include_bias=False)
    if X.ndim == 1:
        # we assume a single sample, in contrast to a single feature
        Xpoly, = pf.fit_transform(X.reshape((1, -1)))
    else:
        Xpoly = pf.fit_transform(X)
    return (np.sign(Xpoly)
            * np.power(np.abs(Xpoly), 1. / np.sum(pf.powers_, axis=-1)))


def make_usefulness(usefulness_scheme: str, min_usefulness: float,
                    max_usefulness: float, n_values: int,
                    tail_power: float = None) -> np.ndarray:
    """
    Make an array of usefulness values according to selected scheme

    :param usefulness_scheme: 'linear', 'exponential', 'longtailed'
    :param min_usefulness: lowest usefulness in output
    :param max_usefulness: highest usefulness in output
    :param n_values: number of usefulness values
    :param tail_power: exponent, for scheme 'longtailed' only
    :return: usefulness values in decreasing order
    """
    if usefulness_scheme == 'linear':
        hidden_usefulness = np.linspace(max_usefulness,
                                        min_usefulness,
                                        n_values)
    elif usefulness_scheme == 'exponential':
        hidden_usefulness = np.logspace(np.log10(max_usefulness),
                                        np.log10(min_usefulness),
                                        n_values)
    elif usefulness_scheme == 'longtailed':
        if tail_power is None:
            raise ValueError('Tail power must be set')
        base = np.linspace(np.power(max_usefulness, 1. / tail_power),
                           np.power(min_usefulness, 1. / tail_power),
                           n_values)
        hidden_usefulness = np.power(base, tail_power)
    else:
        raise ValueError(
            'Unknown usefulness_scheme for distributing usefulness')
    return hidden_usefulness


def generate_hidden_features(
        n_classes: int,
        n_samples_per_class: int,
        n_true_features: int,
        n_fake_features: int,
        min_usefulness: float,
        max_usefulness: float,
        usefulness_scheme: str,
        tail_power: float,
        location_distribution: stats.rv_continuous,
        scale_distribution: stats.rv_continuous,
        sampling_distribution: stats.rv_continuous,
        location_ordering_extent: int,
        location_sharing_extent: int,
        random_state: _RANDOM_STATE_TYPE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a low dimensional hidden feature space

    For parameter description see :fun:`generate_feature_space`

    :return:
      * out_features: np.ndarray
         feature space, shape (n_samples_per_class * n_classes, n_features_out)
      * out_labels: np.ndarray
         labels, i.e., class ids, shape (n_samples_per_class * n_classes, )
      * out_usefulness: np.ndarray
         approximate usefulness of the features, shape (n_features_out, )
      * out_ranges: np.ndarray
         approximate range or amplitude of the features,
         shape (n_features_out, )
    """
    random_state = check_random_state(random_state)
    # todo convert asserts into input validation
    assert 1 <= n_classes
    assert 1 <= n_samples_per_class
    assert 0 <= n_true_features
    assert 0 <= n_fake_features
    assert 1 <= n_true_features + n_fake_features
    assert 0 < min_usefulness <= max_usefulness <= 1
    features = np.empty(
        (n_classes, n_samples_per_class, n_true_features + n_fake_features))
    ranges = np.empty((n_true_features + n_fake_features,))
    usefulness_values = np.concatenate(
        [make_usefulness(usefulness_scheme=usefulness_scheme,
                         min_usefulness=min_usefulness,
                         max_usefulness=max_usefulness,
                         n_values=n_true_features,
                         tail_power=tail_power),
         np.zeros((n_fake_features,))]
    )
    for i, usefulness in enumerate(usefulness_values):
        kwargs = {
            'extent': 10 ** (1 - 2 * np.clip(usefulness, 0, 1)),
            'location_distribution': location_distribution,
            'scale_distribution': scale_distribution,
            'sampling_distribution': sampling_distribution,
            'location_ordering_extent': location_ordering_extent,
            'random_state': random_state
        }
        if 0 < usefulness:
            kwargs['location_sharing_extent'] = location_sharing_extent
        else:
            kwargs['location_sharing_extent'] = n_classes
            kwargs['scale_distribution'] = stats.bernoulli(1)
        ef = EffectiveFeature(n_classes, **kwargs)
        ef.fit()
        ranges[i] = ef.range_
        features[..., i], labels = ef.get_samples(
            n_samples_per_class,
            random_state=random_state)
    features = features.reshape((-1, features.shape[-1])).round(10)
    labels = labels.reshape((-1,))
    return features, labels, usefulness_values, ranges


def blend_features(
        hidden_features: np.ndarray,
        labels: np.ndarray,
        hidden_usefulness_values: np.ndarray,
        hidden_ranges: np.ndarray,
        normalize_by_range: bool,
        n_features_out: int,
        polynomial: Union[bool, int],
        blending_mode: str,
        count_distribution: stats.rv_discrete,
        location_ordering_extent: int,
        location_sharing_extent: int,
        relative_usefulness_content: stats.rv_continuous,
        noise_distribution: stats.rv_continuous,
        random_state: _RANDOM_STATE_TYPE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
    """
    Create a high dimensional feature space from the low-dimensional input

    :return:
      * out_features: np.ndarray
         feature space, shape (n_samples_per_class * n_classes, n_features_out)
      * out_labels: np.ndarray
         labels, i.e., class ids, shape (n_samples_per_class * n_classes, )
      * out_usefulness: np.ndarray
         usefulness measure of features, shape (n_features_out, )
      * out_ranges: np.ndarray
         approximate range or amplitude of the features without added noise,
         shape (n_features_out, )
    """
    random_state = check_random_state(random_state)
    # todo convert asserts into input validation
    assert 1 <= n_features_out
    assert (0 <= location_ordering_extent) or (location_ordering_extent == -1)
    assert 0 <= location_sharing_extent
    if polynomial and (blending_mode == 'logarithmic'):
        warnings.warn('Polynomial features do not add complexity '
                      'to the features resulting from logarithmic blending.')
    hidden_features = hidden_features.reshape((-1, hidden_features.shape[-1]))
    labels = labels.reshape((-1,))

    if normalize_by_range:
        hidden_features = hidden_features / hidden_ranges
        hidden_ranges = np.ones_like(hidden_ranges)

    degree = {True: 2, False: 1}.get(polynomial, polynomial)
    if 2 <= degree:
        hidden_features, hidden_ranges, hidden_usefulness_values = (
            map(lambda x: poly_features(x, degree),
                [hidden_features, hidden_ranges, hidden_usefulness_values]))

    fb = FeatureBlender(count_distribution=count_distribution,
                        alpha=1,
                        n_features_out=n_features_out,
                        blending_mode=blending_mode,
                        random_state=random_state)
    mixed_features = fb.fit_transform(hidden_features)
    mixed_ranges = fb.transform([hidden_ranges], amplitude_like=True)[0]
    mixed_usefulness = fb.transform([hidden_usefulness_values],
                                    amplitude_like=True)[0]
    nb = NoiseBlender(noise_distribution=noise_distribution(
        scale=mixed_ranges / noise_distribution.std()),
        blending_mode=blending_mode,
        random_state=random_state)
    beta = relative_usefulness_content.rvs(size=n_features_out,
                                           random_state=random_state)
    assert np.all(0 <= beta) and np.all(beta <= 1)
    out_features = nb.fit_transform(mixed_features, beta).round(10)
    out_usefulness = nb.transform([mixed_usefulness], amplitude_like=True)[0]
    out_labels = labels
    return out_features, out_labels, out_usefulness, mixed_ranges


def generate_feature_space(
        n_classes: int = 100,
        n_samples_per_class: int = 16,
        n_true_features: int = 30,
        n_fake_features: int = 0,
        n_features_out: int = 10000,
        min_usefulness: float = 0.01,
        max_usefulness: float = 0.9,
        usefulness_scheme: str = 'longtailed',
        tail_power: float = 1.5,
        location_distribution: stats.rv_continuous = stats.norm,
        scale_distribution: stats.rv_continuous = stats.uniform(0.5, 1.0),
        sampling_distribution: stats.rv_continuous = stats.norm,
        location_ordering_extent: int = 0,
        location_sharing_extent: int = 0,
        polynomial: Union[bool, int] = False,
        blending_mode: str = 'linear',
        count_distribution: stats.rv_discrete = stats.randint(5, 11),
        relative_usefulness_content: stats.rv_continuous = stats.uniform(0, 1),
        noise_distribution: stats.rv_continuous = stats.norm,
        random_state: _RANDOM_STATE_TYPE = 137
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray]:
    """
    Simulate a large dimensional feature space

    :param n_classes: number of classes (or labels) of the classification 
      problem
    :param n_samples_per_class: number of samples per class
    :param n_true_features: number of underlying true hidden features, they are
      meant to be useful features
    :param n_fake_features: number of underlying fake hidden features, they are
      meant to be fixed random noise;
      these features are intended to be not informative but due to their
      consistent values used in out_features they may carry information about
      the class (or label); to avoid this either pick 0 or a lot of them
    :param n_features_out: number of visible features to be simulated
    :param blending_mode: "linear" simulates measured features using linear
      combination (cf. central limit theorem), "logarithmic" simulates
      measured features by multiplication (resulting distribution
      approximates log-normal)
    :param min_usefulness: minimum usefulness of true hidden features,
      0 < min_usefulness <= max_usefulness
    :param max_usefulness: maximum usefulness of true hidden features,
      min_usefulness <= max_usefulness <= 1
    :param usefulness_scheme: distribution of usefulness, one of "linear",
      "exponential" and "longtailed"
    :param tail_power: exponent for "longtailed" usefulness_scheme
    :param location_distribution: distribution type of the characteristic
      trait of classes, i.e., the envelop of locations for true features
    :param scale_distribution: frozen pre-parametrized distribution of
      the amplitude of the sampling noise (uncertainty of reproduction),
      i.e., the scale of different samples for the same class (or label) in
      true features
    :param sampling_distribution: distribution type of the uncertainty of
      reproduction, i.e., the noise for different samples from the same
      class (or label) in hidden features
    :param location_ordering_extent: keep segments of locations of given
      block size together in each feature independently, use -1 to use
      exactly the same location order; making this parameter other than zero
      helps to reduce the new information added by each true feature because
      their information gets more redundant; default: 0
    :param location_sharing_extent: make locations shared by multiple classes
      in each feature independently, use 0 to make all locations unique;
      making this parameter other than zero helps to reduce the new
      information added by each true feature because their information gets
      more redundant; default: 0
    :param count_distribution: frozen pre-parametrized distribution of the
      number of hidden features taking part in one specific output feature
    :param polynomial: whether to form polynomial features, you can provide
      the maximum degree as an integer, be aware that the hidden feature space
      internally gets transformed to approx. n_features ** degree features
    :param relative_usefulness_content: frozen pre-parametrized distribution
      for the scale of with uninformative noise,
      given as the part of the output range, 0<=part<1
    :param noise_distribution: distribution type of the additive random noise,
      it is scaled automatically
    :param random_state: seed or RandomState instance, use None to auto-seed,
      default: fixed-seed
    :return:
      * out_features: np.ndarray
         visible feature space,
         shape (n_samples_per_class * n_classes, n_features_out)
      * out_labels: np.ndarray
         labels, i.e., class ids, shape (n_samples_per_class * n_classes, )
      * out_usefulness: np.ndarray
         approximate usefulness of visible features, shape (n_features_out, )
      * out_names: np.ndarray
         ordinal id of the features, shape (n_features_out, )
      * hidden_features: np.ndarray
         hidden feature space, i.e., true and fake features,
         shape (n_samples_per_class * n_classes, n_hidden_features)
      * hidden_usefulness: np.ndarray
         approximate usefulness of hidden features, shape (n_hidden_features, )
    """
    random_state = check_random_state(random_state)

    hidden_features, out_labels, hidden_usefulness_values, hidden_ranges = (
        generate_hidden_features(
            n_classes,
            n_samples_per_class,
            n_true_features,
            n_fake_features,
            min_usefulness,
            max_usefulness,
            usefulness_scheme,
            tail_power,
            location_distribution,
            scale_distribution,
            sampling_distribution,
            location_ordering_extent,
            location_sharing_extent,
            random_state
        )
    )

    out_features, out_labels, out_usefulness, mixed_ranges = (
        blend_features(
            hidden_features,
            out_labels,
            hidden_usefulness_values,
            hidden_ranges,
            True,
            n_features_out,
            polynomial,
            blending_mode,
            count_distribution,
            location_ordering_extent,
            location_sharing_extent,
            relative_usefulness_content,
            noise_distribution,
            random_state
        )
    )

    out_names = np.arange(1, n_features_out + 1)
    return (out_features, out_labels, out_usefulness, out_names,
            hidden_features, hidden_usefulness_values)
