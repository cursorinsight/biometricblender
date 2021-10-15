# BiometricBlender

Python package for generating synthetic data.

## How to install

At least python version 3.7.1 is required. Once you have a working python
(virtual) environment, you can install the package. All the methods below
install necessary dependencies too. A fresh default installation of Anaconda
already contains those dependencies.

### Install current version directly from GitHub

```sh
$ pip install git+https://github.com/cursorinsight/biometricblender.git
```

### Install for development

After cloning the repo, you can install from the folder containing `setup.py`.
The `-e .` or `--editable .` switch makes sure not to copy sources but always
import from the local repo. This is ideal for modifying the code in-place:

```sh
$ pip install -e .
```

Alternatively, where using `develop` instead of `install` is the equivalent
of `-e`:

```sh
$ python setup.py develop
```

### Notes

In the example, we referenced the local repo by the relative path `.` but you
can use absolute path as well.

You can install for the current user with the `--user` switch in the above
command without requiring admin (root) privileges, e.g.

```sh
$ pip install --user -e .
```

but expect parallel use of `--user` and `-e .` to fail due to the presence of
the `pyproject.toml` file, for details see
<https://github.com/pypa/pip/issues/7953>.

If you wish to separate your current install from your global python
configuration then consider creating a virtual environment for the current
install. For details read
<https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>.
Under Unix systems this looks like:

```sh
$ python3 -m venv data_synthesis
$ source data_synthesis/bin/activate
$ pip install -e .
```

## How to use

The package contains a single module called `biometric_blender`.

### biometric_blender

The purpose of the data generated by this package is to establish a test
benchmark for

* multiclass classification
* with a huge number of features
* with nontrivial feature correlations
* with approximate a-priori knowledge about the usefulness of features

Run as

```sh
$ python -m biometric_blender
```

or load by

```py
>>> import biometric_blender
```

Command line options are:

```
  --n-labels N_LABELS   identified labels to simulate (default: 100)
  --n-samples-per-label N_SAMPLES_PER_LABEL
                        samples per label (default: 16)
  --n-base-features N_BASE_FEATURES
                        number of underlying useful features (default: 30)
  --n-noise-features N_NOISE_FEATURES
                        number of underlying fixed random features (default: 0)
  --min-usefulness MIN_USEFULNESS
                        minimum usefulness of base features (default: 0.01)
  --max-usefulness MAX_USEFULNESS
                        maximum usefulness of base features (default: 0.9)
  --usefulness-scheme {linear,exponential,longtailed}
                        distribution of usefulness in base features (default: longtailed)
  --tail-power TAIL_POWER
                        exponent for longtailed usefulness-scheme (default: 1.5)
  --polynomial          use polynomial blending of features (default: False)
  --average-consecutive-locations AVERAGE_CONSECUTIVE_LOCATIONS
                        keep segments of locations of given block size together in each feature
                        independently, use -1 to use exactly the same location order (default: 0)
  --average-shared-locations AVERAGE_SHARED_LOCATIONS
                        make locations shared by multiple labels in each feature independently, use 0
                        to make all locations unique (default: 0)
  --n-features-out N_FEATURES_OUT
                        simulated measured features (default: 10000)
  --blending-mode {linear,logarithmic}
                        how to make simulated measured features (default: linear)
  --random-state RANDOM_STATE
                        integer random seed (default: 137)
  --output OUTPUT       output file name (default: out_data.hdf5)
```

For more  details, run `python -m biometric_blender --help`.
