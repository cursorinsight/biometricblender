import numpy as np

from biometric_blender import main
from fixtures import temppath
from h5py import File, is_hdf5
from os.path import isfile


def test_usage(temppath):
    path = temppath
    args = f"""
            --output {path}
            --n-labels 10
            --n-samples-per-label 3
            --n-features-out 100
            """.split()

    main(args)

    do_test_saved_feature_space(path)


def do_test_saved_feature_space(path):
    assert isfile(path)
    assert is_hdf5(path)

    with File(path, 'r') as data:
        assert 'id' in data.keys()
        assert isinstance(data['id'][()], bytes)

        assert 'hash' in data.keys()
        assert isinstance(data['hash'][()], bytes)

        assert 'created_at' in data.keys()
        assert isinstance(data['created_at'][()], bytes)

        assert 'labels' in data.keys()
        assert data['labels'][()].dtype == np.int64
        assert data['labels'].shape == (30,)
        assert 0 <= data['labels'][0]

        assert 'names' in data.keys()
        assert data['names'][()].dtype == np.int64
        assert data['names'].shape == (100,)
        assert 1 <= data['names'][0]

        assert 'usefulness' in data.keys()
        assert data['usefulness'][()].dtype == np.float64
        assert data['usefulness'].shape == (100,)
        assert 0 <= data['usefulness'][0] <= 1

        assert 'features' in data.keys()
        assert data['features'][()].dtype == np.float64
        assert data['features'].shape == (100, 30)
