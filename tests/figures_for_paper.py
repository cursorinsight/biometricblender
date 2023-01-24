"""
The purpose of this test set is to show how easy or difficult the
generated features are. Results are included in the paper.

@author: Stippinger
"""
import time
from contextlib import contextmanager
from typing import Iterable, Tuple, Dict, List, Any, Union, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.utils import check_random_state
from tqdm import tqdm

from biometric_blender.generator_api import EffectiveFeature


# # #  Gridsearch scores for the table of accuracy  # # #

def get_data_on_the_fly(
        n_labels=100, n_samples_per_label=16, n_true_features=40,
        n_fake_features=160, n_features_out=10000, seed=137
) -> Iterable[Tuple[str, str, Dict[str, Any], Tuple[np.ndarray, ...]]]:
    """
    Generate some test data on the fly:
    true only, hidden only, all output features
    """
    from biometric_blender import generate_feature_space

    kw = dict(n_labels=n_labels,
              count_distribution=stats.randint(5, 11),
              min_usefulness=0.50,
              max_usefulness=0.95,
              n_samples_per_label=n_samples_per_label,
              n_true_features=n_true_features,
              n_fake_features=n_fake_features,
              location_ordering_extent=2,
              location_sharing_extent=3,
              n_features_out=n_features_out,
              blending_mode='logarithmic',
              usefulness_scheme='linear',
              random_state=seed)
    fs = generate_feature_space(**kw)
    tr = fs[4][:, :n_true_features], fs[1], fs[5], fs[3], fs[4], fs[5]
    hd = fs[4], fs[1], fs[5], fs[3], fs[4], fs[5]
    yield '-', 'true', kw, tr
    yield '-', 'hidden', kw, hd
    yield '-', 'full', kw, fs


def make_data(
    fn_base: str = 'screening'
):
    """
    Write some test data: use the default configuration
    """
    import sys
    import runpy
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], f'@{fn_base}.args',
                '--output', f'tmp/{fn_base}.hdf5']
    runpy.run_module("biometric_blender", run_name='__main__')
    sys.argv = saved_argv


def read_data(
        fn: str = 'screening.hdf5'
) -> Iterable[Tuple[str, str, Dict[str, Any], Tuple[np.ndarray, ...]]]:
    """
    Read data from file: if available, true only, hidden only, all features
    """
    import h5py as hdf
    with hdf.File(fn, mode='r') as f:
        myhash = f.get('hash', None)[...]
        kw = dict(f['features'].attrs.items())
        data = (f['features'][...].T, f['labels'][...],
                f['usefulness'][...], f['names'][...],
                None, None)
        if 'hidden_features' in f:
            n_true_features = kw['n_true_features']
            true = (f['hidden_features'][:n_true_features].T, f['labels'][...],
                    f['hidden_usefulness'][:n_true_features],
                    f['names'][:n_true_features], None, None)
            hidden = (f['hidden_features'][...].T, f['labels'][...],
                      f['hidden_usefulness'][...], f['names'][...],
                      None, None)
        else:
            true = None
            hidden = None
    print(f'hash of {fn} is {myhash} with', *kw.items(), sep='\n', end='\n\n')
    if hidden is not None:
        yield fn, 'true', {}, true
        yield fn, 'hidden', {}, hidden
    yield fn, 'full', kw, data


def get_reduction(n_components=None, preset=None, preset_only=False,
                  *, seed=4242) -> Iterable[
    Tuple[str, "sklearn.base.TransformerMixin", int]]:
    """
    Get benchmark reduction algorithms
    """
    # Note: FA rotation requires sklearn version > 0.24
    import sklearn
    assert tuple(map(int, sklearn.__version__.split('.'))) >= (0, 24)
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import FunctionTransformer
    if preset is not None:
        preset = np.asarray(preset)
    for n in np.ravel(n_components):
        if n is None:
            if preset is not None:
                yield 'preset', FunctionTransformer(
                    lambda arr: arr[:, preset], validate=True)
            if preset_only:
                continue
            yield 'none', FunctionTransformer(), n
        else:
            if preset is not None:
                yield 'preset', FunctionTransformer(
                    lambda arr: arr[:, preset[:n_components]], validate=True)
            if preset_only:
                continue
            yield 'kbest', SelectKBest(f_classif, k=n), n
            yield 'pca', PCA(n_components=n, random_state=seed), n
            yield 'fa', FactorAnalysis(n_components=n, rotation='varimax',
                                       random_state=seed), n


def get_classifiers(seed=4242) -> Iterable[
    Tuple[str, "sklearn.base.ClassifierMixin"]]:
    """
    Get benchmark classifiers with their default parameters (no gridsearch)
    """
    # see https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    yield 'knn', KNeighborsClassifier()
    yield 'svm', SVC(random_state=seed)
    yield 'rf', RandomForestClassifier(random_state=seed)
    yield 'rf_david', RandomForestClassifier(
        # no equivalent for n_subfeatures=1000,
        n_estimators=10000,  # n_trees=10000,
        max_samples=0.7, bootstrap=True,  # partial_sampling=0.7,
        min_samples_leaf=2,
        min_samples_split=8,
        min_impurity_decrease=0.1,  # min_purity_increase=0.1
    )


def score_classifiers(fn_base: Union[str, dict] = 'screening',
                      output_fn: Optional[str] = None,
                      n_jobs: int = 2):
    """
    Score benchmark classifiers on the data.
    Not used in the BiometricBlender paper.
    """
    from itertools import product as iterprod
    from sklearn.model_selection import cross_val_score
    result = {}
    if isinstance(fn_base, str):
        data_fn = read_data(f'tmp/{fn_base}.hdf5')
        output_fn = output_fn or f'fig/{fn_base}_scores.csv'
    else:
        data_fn = get_data_on_the_fly(**fn_base)
        output_fn = output_fn or 'fig/onthefly_scores.csv'
    for (red_name, red_obj, red_n), \
        (data_fn, data_kind, data_kw, data_fs) in tqdm(
            iterprod(get_reduction(n_components=None), data_fn),
            desc='data&reduction'):
        (out_features, out_labels, out_usefulness, out_names,
         hidden_features, hidden_usefulness) = data_fs
        simplified_features = red_obj.fit_transform(
            out_features, out_labels)
        for (clf_name, clf_obj) in tqdm(
                get_classifiers(), desc='clf', leave=False):
            name = '_'.join([red_name, str(red_n),
                             clf_name, data_kind])
            score = cross_val_score(clf_obj, simplified_features,
                                    out_labels, n_jobs=n_jobs)
            result[name] = score
            print(name, score, flush=True)
            # output after each round
            df = pd.DataFrame(result)
            df.to_csv(output_fn)


def score_screening(  # todo score_gridsearch_screening
        fn_base: str = 'screening',
        importances_fn: str = 'feature_names.txt',
        output_fn: Optional[str] = None,
        n_components: Sequence[int] = (10, 25, 50, 100, 200, 400, 800),
        n_jobs: int = 2):
    """
    Score classifiers on some screened data. Screening results must be
    provided externally. First feature id is 1 opposed to pythonic 0.
    """
    from sklearn.model_selection import cross_val_score
    output_fn = output_fn or f'{fn_base}_screened_scores.csv'
    result = {}
    data_fn, data_kind, data_kw, data_fs = list(
        read_data(f'tmp/{fn_base}.hdf5'))[-1]
    (out_features, out_labels, out_usefulness, out_names,
     hidden_features, hidden_usefulness) = data_fs
    importances = pd.read_csv(
        importances_fn, sep=';', header=None, names=['feature', 'count'])
    for red_n in tqdm(n_components, desc='reduction'):
        important_features = out_features[:, importances.values[:red_n, 0] - 1]
        for (clf_name, clf_obj) in tqdm(
                get_classifiers(), desc='clf', leave=False):
            name = '_'.join([str(red_n), clf_name, data_kind])
            score = cross_val_score(clf_obj, important_features,
                                    out_labels, n_jobs=n_jobs)
            result[name] = score
            print(name, score, flush=True)
            # output after each round
            df = pd.DataFrame(result)
            df.to_csv(output_fn)


def get_gridsearch_classifiers(seed=4242) -> Iterable[
    Tuple[str, object, Dict[str, list]]]:
    """
    Get benchmark classifiers to test with various parametrization
    """
    # see https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    yield 'knn', KNeighborsClassifier(), {
        "weights": ['uniform', 'distance'],
    }
    yield 'svm', SVC(random_state=seed), {
        "C": [0.5, 1.0, 2.0],
        "tol": [1e-4, 1e-3, 1e-2],
    }
    yield 'rf', RandomForestClassifier(random_state=seed), {
        "n_estimators": [1000],
        "min_samples_leaf": [1, 2, 4],
        "min_impurity_decrease": [0.0, 0.01, 0.05],
        "max_depth": [None, 8, 10],
    }


def score_gridsearch_classifiers(
        fn_base: Union[str, dict] = 'screening',
        output_fn: Optional[str] = None,
        n_components: Sequence[int] = (None, 10, 25, 50, 100, 200, 400, 800),
        n_jobs: int = 4):
    """
    Score benchmark classifiers with various parametrization on the data
    """
    from itertools import product as iterprod
    from sklearn.model_selection import GridSearchCV
    result = []
    if isinstance(fn_base, str):
        data = read_data(f'tmp/{fn_base}.hdf5')
        output_fn = output_fn or f'fig/{fn_base}_gridsearch_scores.csv'
    else:
        data = get_data_on_the_fly(**fn_base)
        output_fn = output_fn or 'fig/onthefly_gridsearch_scores.csv'
    for (red_name, red_obj, red_n),\
        (data_fn, data_kind, data_kw, data_fs) in tqdm(
            iterprod(get_reduction(n_components=n_components), data),
            desc='data&reduction'):
        (out_features, out_labels, out_usefulness, out_names,
         hidden_features, hidden_usefulness) = data_fs
        if (red_n is not None) and (out_features.shape[1] < red_n):
            continue
        t0 = time.time()
        simplified_features = red_obj.fit_transform(
            out_features, out_labels)
        red_time = time.time() - t0
        for (clf_name, clf_obj, clf_param_grid) in tqdm(
                get_gridsearch_classifiers(), desc='clf', leave=False):
            gridsearch = GridSearchCV(clf_obj, clf_param_grid, cv=4,
                                      verbose=2, n_jobs=n_jobs)
            gridsearch.fit(simplified_features, out_labels)
            df = pd.DataFrame(gridsearch.cv_results_)
            df['reduction'] = red_name
            df['reduction_time'] = red_time
            df['n_components'] = red_n
            df['classifier'] = clf_name
            df['data_fn'] = data_fn
            df['data_kind'] = data_kind
            result.append(df)
            # output after each round
            pd.concat(result).to_csv(output_fn)


def make_table_accuracy(fn_base: str, data_kind: str):
    """
    Find the best parametrization from stored scores
    (write out TeX tables presented in the paper)
    """
    df = pd.read_csv(f'{fn_base}_gridsearch_scores.csv')
    outcome = df.sort_values(
        'mean_test_score', ascending=False
    ).drop_duplicates(
        ['data_kind', 'classifier', 'reduction', ]
    )
    q = f"data_kind=='{data_kind}'"
    tmp = outcome.query(q).set_index(['classifier', 'reduction'])
    columns = ['none', 'pca', 'fa', 'kbest']
    rows = ['knn', 'svm', 'rf']
    new_columns = {'pca': 'PCA', 'fa': 'FA', 'kbest': '$k$-best'}
    new_rows = {'knn': '$k$NN', 'svm': 'SVC', 'rf': 'RF'}
    tmp.loc[:, 'mean_test_score'].unstack('reduction').round(3).reindex(
        index=rows, columns=columns).rename(
        index=new_rows, columns=new_columns).to_latex(
        f'{fn_base}-score-{data_kind}.tex')
    tmp.loc[:, 'mean_fit_time'].unstack('reduction').reindex(
        index=rows, columns=columns).rename(
        index=new_rows, columns=new_columns).to_latex(
        f'{fn_base}-time-fit-{data_kind}.tex')
    tmp.loc[:, 'reduction_time'].unstack('reduction').reindex(
        index=rows, columns=columns).rename(
        index=new_rows, columns=new_columns).to_latex(
        f'{fn_base}-time-red-{data_kind}.tex')
    pass


def make_figure_accuracy(fn_base: str, data_kind: str):
    """
    Make figure from stored scores as a function of n_components
    (from the various parametrizations only the best score is kept)
    """
    from matplotlib import pyplot as plt
    df = pd.read_csv(f'{fn_base}_gridsearch_scores.csv')
    outcome = df.sort_values(
        'mean_test_score', ascending=False
    ).drop_duplicates(
        ['data_kind', 'classifier', 'reduction', 'n_components', ]
    )
    outcome.to_excel(f'{fn_base}_outcome.xlsx')
    reduction = list(o for o in outcome.reduction.unique() if o != 'none')
    if not len(reduction):
        reduction = ['none']
    fig, ax = plt.subplots(3, len(reduction),
                           sharex=True, sharey='row', squeeze=False)
    for i, red in enumerate(reduction):
        ax[0, i].set_title(red)
        ax[0, i].semilogx()
        for clf in outcome.classifier.unique():
            q = "reduction=='{}' & classifier=='{}' & data_kind=='{}'".format(
                red, clf, data_kind)
            meas = outcome.query(q).sort_values('n_components')
            q = "reduction=='{}' & classifier=='{}' & data_kind=='{}'".format(
                'none', clf, data_kind)
            ref = outcome.query(q).iloc[0, :]
            # top row: score
            l0, = ax[0, i].plot(meas['n_components'],
                                meas['mean_test_score'],
                                marker='o',
                                markersize='3',
                                markerfacecolor='w',
                                markeredgewidth=0.5,
                                label=clf)
            lr = ax[0, i].axhline(ref['mean_test_score'],
                                  color=l0.get_color(),
                                  linestyle='--')
            # middle row: fit time
            l1, = ax[1, i].plot(meas['n_components'],
                                meas['mean_fit_time'],
                                marker='o',
                                markersize='3',
                                markerfacecolor='w',
                                markeredgewidth=0.5,
                                label=clf)
            lt = ax[1, i].axhline(ref['mean_fit_time'],
                                  color=l1.get_color(),
                                  linestyle='--')
            # bottom row: reduction time
            l2, = ax[2, i].plot(meas['n_components'],
                                meas['reduction_time'],
                                marker='o',
                                markersize='3',
                                markerfacecolor='w',
                                markeredgewidth=0.5,
                                label=clf)
            lr = ax[2, i].axhline(ref['reduction_time'],
                                  color=l2.get_color(),
                                  linestyle='--')
        # add legend entry
        ll, = ax[0, i].plot([np.nan], [np.nan],
                            color='k',
                            linestyle='--',
                            label='no red.')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, title='gridsearch\nclassifier')
    ax[0, 0].set_ylabel('max(accuracy)')
    ax[1, 0].set_ylabel('fit time')
    ax[2, 0].set_ylabel('reduction time')
    ax[-1, 0].set_xlabel('reduction n_components')
    fig.savefig(f'{fn_base}_figure.pdf')
    plt.show()


# # #  Additional figure about the reconstruction capabilities of FA  # # #

def compute_scores_for_n_components(X, red):
    """
    Cross validated reduction scores for varying n_components.

    Notes: This could be a GridSearchCV. This function is used only by
    plot_factor_analysis_reconstruction just below.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone
    red = clone(red)
    n_components = np.logspace(0, np.log10(np.minimum(X.shape[1], 200)),
                               num=10)
    n_components = np.unique(n_components.astype(int))
    scores = []
    for n in tqdm(n_components):
        red.n_components = n
        scores.append(np.mean(cross_val_score(red, X, cv=3)))

    return n_components, scores


def plot_factor_analysis_reconstruction():
    """
    Estimate number of factors based on cross-validated model likelihood.
    Plot a matrix of original vs varimax rotated inferred factors.

    Notes: This is an exploratory figure not included in the BiometricBlender
    paper. It uses the entire matrix output of FA, therefore calculation and
    plotting are bundled.
    """
    from sklearn.decomposition import FactorAnalysis
    from scipy.stats import spearmanr
    for name, kw, fs in get_data_on_the_fly(n_fake_features=40):
        (out_features, out_labels, out_usefulness, out_names,
         hidden_features, hidden_usefulness) = fs
        sorter = np.argsort(hidden_usefulness)[::-1]  # decreasing
        ranked_usefulness = hidden_usefulness[sorter]
        ranked_hidden_features = hidden_features[:, sorter]
        fa = FactorAnalysis(rotation='varimax')
        n_hidden = hidden_features.shape[1]
        n_components, scores = compute_scores_for_n_components(out_features,
                                                               fa)
        n_ml = n_components[np.argmax(scores)]
        fa.n_components = n_ml
        reconstructred = fa.fit_transform(out_features, out_labels)
        print(out_features.shape, reconstructred.shape)
        corr_result = spearmanr(ranked_hidden_features, reconstructred)
        reconstruction_corr = corr_result.correlation[:n_hidden, n_hidden:]
        corr_result = spearmanr(ranked_hidden_features, out_features)
        out_corr = corr_result.correlation[:n_hidden, n_hidden:]

        fig, ax = plt.subplots(2, 2,
                               figsize=(8, 6))  # type: plt.Figure, plt.Axes
        ax = ax.ravel()  # type: list[plt.Axes]
        ax[3].invert_yaxis()
        ax[2].get_shared_y_axes().join(ax[3])

        h0 = ax[1].hist(out_corr.max(axis=0))
        ax[1].semilogy()
        ax[1].set_xlabel('max correlation to any hidden feature')
        ax[1].set_ylabel('# output features')
        l1, = ax[0].plot(n_components, scores, marker='o')
        ax[0].semilogx()
        ax[0].set_xlabel('n_components')
        ax[0].set_ylabel('likelihood')
        mx = ax[2].matshow(np.abs(reconstruction_corr), vmin=0, vmax=1)
        ax[2].set_xlabel('reconstructed')
        ax[2].set_ylabel('original')
        plt.colorbar(mx, ax=ax[3])
        l2u, = ax[3].plot(ranked_usefulness, np.arange(n_hidden),
                          label='usefulness')
        f2u = ax[3].fill_betweenx(np.arange(n_hidden), 0,
                                  ranked_usefulness, alpha=0.4,
                                  color=l2u.get_color())
        sac = np.max(np.abs(reconstruction_corr), axis=1)
        l2c, = ax[3].plot(sac, np.arange(n_hidden), label='max abs corr')
        f2c = ax[3].fill_betweenx(np.arange(n_hidden), 0, sac,
                                  alpha=0.4, color=l2c.get_color())
        ax[3].set_xlabel('usefulness or detectability')
        ax[3].set_ylabel('rank')
        ax[3].legend()
        fig.savefig('fig/fa_{}.pdf'.format(name))
        plt.show()


# # #  Figures for the targeted usefulness of hidden features  # # #

@contextmanager
def intercept_ef():
    """
    Hack to get parametrization of EffectiveFeatures within a context
    """
    from biometric_blender import generator_api
    original = generator_api.EffectiveFeature
    instances = []

    class Replacement(generator_api.EffectiveFeature):
        def get_samples(self, *args, **kwargs):
            instances.append(self)
            return super(Replacement, self).get_samples(*args, **kwargs)

    generator_api.EffectiveFeature = Replacement
    generator_api.patched = True
    try:
        yield instances
    finally:
        # to check: do we restore original state under all circumstances
        generator_api.EffectiveFeature = original
        del generator_api.patched


def plot_1d_locations(
        ax: plt.Axes,
        ef: EffectiveFeature,
        reverse: bool,
        normalize: bool
):
    def dist_pdf(dist, shift=0., **kwargs):
        xr = dist.mean() + np.array([-4, 4]) * dist.std()
        x = np.linspace(*xr, 40)
        y = dist.pdf(x)
        if normalize:
            y = y / np.max(y) + shift
        if reverse:
            ax.plot(y, x, **kwargs)
        else:
            ax.plot(x, y, **kwargs)

    shift = 0.
    for i, (loc, scale) in enumerate(zip(ef.locations_, ef.scales_)):
        dist = ef.sampling_distribution(loc, scale)
        dist_pdf(dist, linestyle='-', shift=shift * i)
    dist_pdf(ef.location_distribution, shift=shift * len(ef.locations_),
             color='k', linestyle='--')


def plot_2d_realizations(ax: plt.Axes, fs: np.ndarray, labels: np.ndarray):
    df = pd.DataFrame(fs,
                      index=pd.Index(labels, name='labels'),
                      columns=['x', 'y'])
    for i, data in df.groupby('labels'):
        ax.plot(data.x, data.y,
                marker='o', markersize=2, linestyle='none')


def make_features_by_usefulness(
        seed: int = 137,
        usefulness: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[EffectiveFeature]]:
    from scipy import stats
    from biometric_blender import generator_api
    rs = check_random_state(seed)
    with intercept_ef() as instances:
        fs, labels, _, _ = generator_api.generate_hidden_features(
            10, 16, 2, 0, usefulness, usefulness, 'linear', None, stats.norm,
            stats.uniform(0.5, 1.5), stats.norm, 2, 2, rs)
    return fs, labels, instances


def make_slides_usefulness_demo(seed=137):
    """
    Show the effect of usefulness on two features and their locations
    (each usefulness is saved to a separate figure like a slideshow)
    """
    for i, usefulness in enumerate([0.01, 0.1, 0.3, 0.5, 0.99]):
        fs, labels, instances = make_features_by_usefulness(
            seed=seed, usefulness=usefulness
        )

        fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
        plot_1d_locations(ax[0, 0], instances[0],
                          reverse=False, normalize=False)
        plot_1d_locations(ax[1, 1], instances[1],
                          reverse=True, normalize=False)
        plot_2d_realizations(ax[1, 0], fs, labels)
        ax[0, 1].remove()
        ax[1, 0].set_xlabel('feature A')
        ax[1, 0].set_ylabel('feature B')

        fig.suptitle(f'usefulness={usefulness}')
        fig.savefig(f'fig/usefulness-zoom-{i}.png')

        ax[1, 0].set_xlim([-30, 30])
        ax[1, 0].set_ylim([-30, 30])
        fig.suptitle(f'usefulness={usefulness}')
        fig.savefig(f'fig/usefulness-fixed-{i}.png')


def make_figure_usefulness_demo(seed=137):
    """
    Show the effect of usefulness on two features
    (save the figure presented in the paper)
    """

    def get_mnl_top():
        return MaxNLocator(nbins=1, integer=True,
                           symmetric=False, min_n_ticks=2)

    def get_mnl_bottom():
        return MaxNLocator(nbins=2, integer=True,
                           symmetric=True, min_n_ticks=3)

    fig, ax = plt.subplots(2, 3, figsize=(5, 3),
                           gridspec_kw={'wspace': 0.3},
                           sharex='col', sharey=False)
    for i, usefulness in enumerate([0.2, 0.4, 0.6]):
        fs, labels, instances = make_features_by_usefulness(
            seed=seed, usefulness=usefulness
        )

        plot_1d_locations(ax[0, i], instances[0],
                          reverse=False, normalize=False)
        plot_2d_realizations(ax[1, i], fs, labels)
        ax[0, i].update_datalim([[0, 0], [0, 1]])
        ax[0, i].yaxis.set_major_locator(get_mnl_top())
        ax[1, i].xaxis.set_major_locator(get_mnl_bottom())
        ax[1, i].yaxis.set_major_locator(get_mnl_bottom())
        ax[0, i].set_title(f'usefulness={usefulness}')
    ax[0, 0].set_ylabel('pdf of A')
    ax[1, 0].set_xlabel('feature A')
    ax[1, 0].set_ylabel('feature B')
    fig.align_ylabels(ax[:, 0])

    fig.savefig(f'fig/usefulness-autozoom.png', bbox_inches='tight')
    fig.savefig(f'fig/usefulness-autozoom.pdf', bbox_inches='tight')


# # #  Entry point  # # #

def main(usefulness_demo: bool,
         generate: bool,
         fn_base: str,
         *,
         gridsearch: bool = True,
         importances_fn: Optional[str] = None,
         output_fn: Optional[str] = None,
         ):
    # # prerequisites # #
    import os
    os.makedirs('fig', exist_ok=True)
    os.makedirs('tmp', exist_ok=True)

    if generate:
        print('generating...', flush=True)
        make_data(fn_base)

    # # simple scoring # #
    # score_classifiers(fn_base, output_fn=None, n_jobs=4)

    if importances_fn:
        # Importances must be provided externally.
        score_screening(fn_base, importances_fn=importances_fn,
                        output_fn=output_fn)

    # # gridsearch # #
    if gridsearch:
        print('scoring takes a while...', flush=True)
        score_gridsearch_classifiers(fn_base, output_fn=None, n_jobs=2)
        print('summarizing...', flush=True)
        for data_kind in ['true', 'hidden', 'full']:
            make_table_accuracy(fn_base, data_kind)
            make_figure_accuracy(fn_base, data_kind)

    # # supplementary # #
    if usefulness_demo:
        make_slides_usefulness_demo()
        make_figure_usefulness_demo()


def cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--usefulness-demo', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--gridsearch', action='store_true')
    parser.add_argument('--fn_base', type=str, default='blending')
    parser.add_argument('--importances_fn', type=str, default=None)
    parser.add_argument('--output_fn', type=str, default=None)  # screening!
    args = parser.parse_args()
    main(usefulness_demo=args.usefulness_demo,
         generate=args.generate, gridsearch=args.gridsearch,
         fn_base=args.fn_base, importances_fn=args.importances_fn,
         output_fn=args.output_fn)


if __name__ == '__main__':
    cli()
