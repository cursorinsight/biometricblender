"""
The purpose of this test set is to test all aspects of feature generation.

:author: Stippinger
"""

from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MaxNLocator


class TestSegmentShuffle(TestCase):
    def test_get_block_sizes(self):
        from biometric_blender.generator_api import SegmentShuffle
        from scipy import stats
        from sklearn.utils import check_random_state
        total = 55
        for par in [7, 4.5, stats.uniform(7, 10)]:
            segs = SegmentShuffle(par).get_block_sizes(
                total,
                random_state=check_random_state(137))
            self.assertEqual(np.sum(segs), total)
        par = 20
        for total in range(79, 121):
            segs = SegmentShuffle(par).get_block_sizes(
                total,
                random_state=check_random_state(137))
            self.assertEqual(np.sum(segs), total)
            self.assertEqual(len(segs), total // par)

    def test_transform(self):
        from biometric_blender.generator_api import SegmentShuffle
        from scipy import stats
        from sklearn.utils import check_random_state
        total = 1023
        for par in [7, 4.5, stats.uniform(7, 10)]:
            segs = SegmentShuffle(par, random_state=check_random_state(
                137)).fit_transform(np.arange(total) + 1)
            self.assertEqual(len(np.unique(segs)), total)


class TestEffectiveFeature(TestCase):
    def test_get_samples(self):
        from biometric_blender.generator_api import EffectiveFeature
        n_classes = 20
        n_samples_per_class = 100
        ef = EffectiveFeature(n_classes, 0.2, random_state=137)
        ef.fit()
        samples, labels = ef.get_samples(
            n_samples_per_class=n_samples_per_class)
        self.assertEqual(samples.shape, labels.shape)
        self.assertEqual(samples.shape, (n_classes, n_samples_per_class))
        return samples

    def test_sample_extent(self):
        from biometric_blender.generator_api import EffectiveFeature
        n_classes = 20
        n_extents = 100
        samples = np.empty((n_classes, n_extents))
        for i in range(n_extents):
            ef = EffectiveFeature(
                n_classes, 10 ** (float(i) / n_extents - 0.5),
                random_state=i, location_ordering_extent=-1)
            ef.fit()
            samples[:, i:i + 1], labels = ef.get_samples(n_samples_per_class=1)
        return samples


class TestFeatureBlender(TestCase):
    def test_transform(self):
        from biometric_blender.generator_api import FeatureBlender
        data = np.random.uniform(0, 1, size=(100, 11))
        n_features_out = 13
        fm = FeatureBlender(n_features_out=n_features_out)
        out = fm.fit_transform(data)
        self.assertEqual(out.shape, (100, 13))

    def test_transform_iris(self):
        from sklearn.datasets import load_iris
        from biometric_blender.generator_api import FeatureBlender
        iris = load_iris()
        n_features_out = 103
        fm = FeatureBlender(n_features_out=n_features_out)
        out = fm.fit_transform(iris['data'], iris['target'])
        expect = len(iris['data']), n_features_out
        self.assertEqual(out.shape, expect)

    def test_transform_wine(self):
        from sklearn.datasets import load_wine
        from biometric_blender.generator_api import FeatureBlender
        wine = load_wine()
        n_features_out = 1013
        fm = FeatureBlender(n_features_out=n_features_out)
        out = fm.fit_transform(wine['data'], wine['target'])
        expect = len(wine['data']), n_features_out
        self.assertEqual(out.shape, expect)


class TestGenerator(TestCase):
    def test_make_usefulness(self):
        from biometric_blender.generator_api import make_usefulness
        self.assertTrue(
            np.allclose(make_usefulness('linear', 0.1, 0.9, 5).tolist(),
                        [0.9, 0.7, 0.5, 0.3, 0.1]))
        self.assertTrue(
            np.allclose(make_usefulness('exponential', 0.2, 0.8, 3).tolist(),
                        [0.8, 0.4, 0.2]))
        self.assertTrue(np.allclose(
            make_usefulness('longtailed', 0.01, 0.81, 5, 2).tolist(),
            [0.81, 0.49, 0.25, 0.09, 0.01]))

    @staticmethod
    def generate_some(n_classes=100,
                      n_samples_per_class=16,
                      n_true_features=9,
                      n_fake_features=93,
                      n_features_out=1013
                      ):
        from biometric_blender import generate_feature_space
        from scipy.stats import randint, uniform

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  min_usefulness=1,
                  max_usefulness=1,
                  n_fake_features=0)
        fs = generate_feature_space(**kw)
        yield 'all useful', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  min_usefulness=0.1,
                  max_usefulness=1,
                  n_fake_features=0)
        fs = generate_feature_space(**kw)
        yield 'basic', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  polynomial=True)
        fs = generate_feature_space(**kw)
        yield 'polynomial', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  relative_usefulness_content=uniform(0.6, 0.4),
                  blending_mode='logarithmic')
        fs = generate_feature_space(**kw)
        yield 'logarithmic', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  blending_mode='logarithmic',
                  relative_usefulness_content=randint(1, 2))
        fs = generate_feature_space(**kw)
        yield 'noiseless logarithmic', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  n_fake_features=0,
                  relative_usefulness_content=uniform(0.6, 0.4),
                  blending_mode='logarithmic')
        fs = generate_feature_space(**kw)
        yield 'true logarithmic', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  n_fake_features=0,
                  blending_mode='logarithmic',
                  relative_usefulness_content=randint(1, 2))
        fs = generate_feature_space(**kw)
        yield 'pure logarithmic', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  location_ordering_extent=3)
        fs = generate_feature_space(**kw)
        yield 'ordered', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  location_sharing_extent=3)
        fs = generate_feature_space(**kw)
        yield 'shared', kw, fs

        kw = dict(n_classes=n_classes,
                  n_samples_per_class=n_samples_per_class,
                  n_true_features=n_true_features,
                  n_features_out=n_features_out,
                  n_fake_features=n_fake_features)
        fs = generate_feature_space(**kw)
        yield 'noise', kw, fs

    def test_generate_feature_space(self):
        for name, kw, fs in self.generate_some():
            out_features, out_labels, out_usefulness, out_names, _, _ = fs
            self.assertEqual(out_features.shape, (
                kw['n_samples_per_class'] * kw['n_classes'],
                kw['n_features_out']))
            self.assertEqual(out_labels.shape,
                             (kw['n_samples_per_class'] * kw['n_classes'],))
            self.assertEqual(out_usefulness.shape, (kw['n_features_out'],))
            self.assertEqual(out_names.shape, (kw['n_features_out'],))

    def test_usefulness_vs_importance(self):
        from scipy.stats import rankdata
        from biometric_blender.tree_analysis import \
            forest_decision_feature_importance, RandomForestClassifier

        for name, kw, fs in self.generate_some():
            out_features, out_labels, out_usefulness, out_names, _, _ = fs
            rf = RandomForestClassifier()
            rf.fit(out_features, out_labels)
            fi = forest_decision_feature_importance(rf)

            out_rank = rankdata(out_usefulness)
            fi_rank = rankdata(fi)
            corr = np.corrcoef(out_rank, fi_rank)[0, 1]

            # In general, due to mixing and noise we do not expect the very
            # same shape of the out_usefulness anf fi curves but ranks should
            # match to a certain level
            msg = f"Usefulness of {name} did not match forest-based importance"
            self.assertGreater(corr, 0.3, msg=msg)


class TestMain(TestCase):
    def test_main(self):
        import sys
        import runpy
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], '--output', 'fig/out_main_call.h5']
        runpy.run_module("biometric_blender", run_name='__main__')
        sys.argv = saved_argv

    def test_cli(self):
        import sys
        import runpy
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], '--no-such-argument',
                    '--output', 'fig/out_raises.h5']
        self.assertRaises(SystemExit, runpy.run_module, "biometric_blender",
                          run_name='__main__')
        sys.argv = [saved_argv[0], '--polynomial',
                    '--output', 'fig/out_poly.h5']
        runpy.run_module("biometric_blender", run_name='__main__')
        sys.argv = [saved_argv[0], '--location-ordering-extent', '5',
                    '--store-hidden', '--output', 'fig/out_ordering.h5']
        runpy.run_module("biometric_blender", run_name='__main__')
        sys.argv = saved_argv


class VisualInspection(TestCase):
    def test_get_samples(self):
        samples = TestEffectiveFeature().test_get_samples()
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        ax.plot(samples.T)
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('sample id')
        ax.set_ylabel('feature value')
        plt.show(block=False)
        plt.pause(1)  # give time to draw the figure

    def test_feature_blender(self):
        from itertools import product as iterprod
        from biometric_blender import FeatureBlender
        fig, ax = plt.subplots(2, 1, figsize=(
            4, 8))  # type: plt.Figure, list[plt.Axes]
        a = np.array([-2, 0., 0.5, 1., 2.])
        b = np.linspace(-1, 1, 100)
        c = [(x, y) for x, y in iterprod(a, a) if x < y]
        w = np.stack([b, 1 - np.abs(b)], axis=-1)
        fb = FeatureBlender()
        fb.n_features_in_ = 2
        fb.weights_ = w
        out = fb.transform(c)
        ax[0].plot(b, out.T)
        fb.blending_mode = 'logarithmic'
        out = fb.transform(c)
        ax[1].plot(b, out.T)
        plt.show()
        pass

    def test_noise_blender(self):
        from biometric_blender import NoiseBlender
        fig, ax = plt.subplots(2, 1, figsize=(
            4, 8))  # type: plt.Figure, list[plt.Axes]
        a = np.repeat([-2, 0., 0.5, 1., 2.], 2)[:, np.newaxis]
        w = np.linspace(0, 1, 100)
        nb = NoiseBlender()
        nb.n_features_in_ = 1
        nb.weights_ = w
        out = nb.transform(a)
        ax[0].plot(w, out.T)
        nb.blending_mode = 'logarithmic'
        out = nb.transform(a)
        ax[1].plot(w, out.T)
        plt.show()
        pass

    def test_usefulness_vs_importance(self):
        from biometric_blender.tree_analysis import \
            forest_decision_feature_importance, RandomForestClassifier
        for name, kw, fs in TestGenerator().generate_some():
            if 'log' not in name:
                continue
            out_features, out_labels, out_usefulness, out_names, _, _ = fs
            rf = RandomForestClassifier()
            rf.fit(out_features, out_labels)
            fi = forest_decision_feature_importance(rf)

            fig, ax = plt.subplots(2, 1, figsize=(
                4, 8))  # type: plt.Figure, list[plt.Axes]
            l0, = ax[0].plot(np.sort(out_usefulness),
                             np.arange(len(out_usefulness)),
                             label='usefulness parameter',
                             color='tab:orange', linestyle='None', marker='o',
                             markersize=2)
            axt = ax[0].twiny()
            l1, = axt.plot(np.sort(fi),
                           np.arange(len(out_usefulness)),
                           label='forest-based importance',
                           color='tab:purple', linestyle='None', marker='o',
                           markersize=2)
            fig.suptitle(name)
            ax[0].set_ylabel('rank')
            ax[0].set_xlabel('usefulness parameter')
            axt.set_xlabel('forest-based importance')
            ax[0].legend(title='cdf', handles=[l0, l1], loc='lower right')
            ax[1].plot(out_usefulness, fi, linestyle='None', marker='o',
                       markersize=2)
            ax[1].set_xlabel('usefulness parameter')
            ax[1].set_ylabel('forest-based importance')
            fig.savefig("fig/{}.pdf".format(name), bbox_inches='tight')
            plt.show()
