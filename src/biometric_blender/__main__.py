"""
The purpose of this module is to generate a plausible random feature set

:author: Stippinger
"""

import argparse
import uuid
from datetime import datetime

import h5py as hdf

from .generator_api import generate_feature_space, stats

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='BiometricBlender: Ultra-high dimensional, multi-class '
                    'synthetic data generator to imitate biometric feature '
                    'space',
        epilog='Copyright (C) The BiometricBlender contributors, 2021.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    p.add_argument('--n-labels',
                   help='identified labels (classes) to simulate',
                   default=100, type=int)
    p.add_argument('--n-samples-per-label',
                   help='samples per label (class)',
                   default=16, type=int)
    p.add_argument('--n-true-features',
                   help='number of underlying true hidden features, they are '
                        'meant to be useful features',
                   default=40, type=int)
    p.add_argument('--n-fake-features',
                   help='number of underlying fake hidden features, they are '
                        'meant to be fixed random noise',
                   default=0, type=int)
    p.add_argument('--min-usefulness',
                   help='minimum usefulness of true hidden features',
                   default=0.50, type=float)
    p.add_argument('--max-usefulness',
                   help='maximum usefulness of true hidden features',
                   default=0.95, type=float)
    p.add_argument('--usefulness-scheme',
                   help='distribution of usefulness in true hidden features',
                   default='linear', type=str,
                   choices=['linear', 'exponential', 'longtailed'])
    p.add_argument('--tail-power',
                   help='exponent for longtailed usefulness-scheme',
                   default=1.5, type=float)
    p.add_argument('--location-distribution',
                   help='distribution type of the characteristic trait of '
                        'labels (classes), i.e., the envelop of locations '
                        'for true features',
                   default='norm', choices=['norm', 'uniform'], type=str)
    p.add_argument('--sampling-distribution',
                   help='distribution type of the uncertainty of reproduction,'
                        'i.e., the noise for different samples from the same '
                        'label (class) in hidden features',
                   default='norm', choices=['norm', 'uniform'], type=str)
    p.add_argument('--location-ordering-extent',
                   help='keep segments of locations of given block size '
                        'together in each feature independently, use -1 to '
                        'use exactly the same location order',
                   default=0, type=int)
    p.add_argument('--location-sharing-extent',
                   help='make locations shared by multiple labels (classes) '
                        'in each feature independently, use 0 to make all '
                        'locations unique',
                   default=0, type=int)
    p.add_argument('--polynomial',
                   help='use polynomial mixing of features',
                   default=False, action='store_true')
    p.add_argument('--n-features-out',
                   help='number of measured features to be simulated',
                   default=10000, type=int)
    p.add_argument('--blending-mode',
                   help='how to simulate measured features',
                   default='linear',
                   choices=['linear', 'logarithmic'], type=str)
    p.add_argument('--min-count',
                   help='minimum number of hidden features taking part in one '
                        'specific output feature',
                   default=5, type=int)
    p.add_argument('--max-count',
                   help='maximum number of hidden features taking part in one '
                        'specific output feature',
                   default=10, type=int)
    p.add_argument('--min-noise',
                   help='minimum noise of output features',
                   default=0.00, type=float)
    p.add_argument('--max-noise',
                   help='maximum noise of output features',
                   default=1.00, type=float)
    p.add_argument('--store-hidden',
                   help='store the hidden feature space for later analysis',
                   action='store_true')
    p.add_argument('--random-state',
                   help='integer random seed',
                   default=137, type=int)
    p.add_argument('--output',
                   help='output file name',
                   default='out_data.hdf5', type=argparse.FileType(mode='wb'))
    argspace = p.parse_args()
    generate_args = vars(argspace)
    output_file = generate_args.pop('output')
    store_hidden = generate_args.pop('store_hidden')
    generate_args_original_copy = generate_args.copy()
    for arg_name in ['location_distribution', 'sampling_distribution']:
        try:
            dist = getattr(stats, generate_args[arg_name])
        except AttributeError:
            msg = 'Unknown distribution "{}"'.format(generate_args[arg_name])
            raise ValueError(msg) from None
        else:
            generate_args[arg_name] = dist
    min_count = generate_args.pop('min_count')
    max_count = generate_args.pop('max_count')
    if max_count < min_count:
        msg = ('min_count must be less or equal than max_count, got {} and {}'
               'respectively'.format(min_count, max_count))
        raise ValueError(msg)
    generate_args['count_distribution'] = (
        stats.randint(min_count, max_count + 1))
    min_noise = generate_args.pop('min_noise')
    max_noise = generate_args.pop('max_noise')
    if max_noise < min_noise:
        msg = ('min_noise must be less or equal than max_noise, got {} and {}'
               'respectively'.format(min_noise, max_noise))
        raise ValueError(msg)
    generate_args['relative_usefulness_content'] = (
        stats.uniform(1 - max_noise, (1 - min_noise) - (1 - max_noise)))
    (out_features, out_labels, out_usefulness, out_names,
     hidden_features, hidden_usefulness) = (
        generate_feature_space(**generate_args))
    output_file.close()  # close because hdf5 requires file name, not object
    try:
        import hashlib
        # Note: python hash function is salted
        m = hashlib.sha256()
        # Note: converting to bytes increases memory need
        m.update(out_features.tobytes())
        m.update(out_labels.tobytes())
        my_hash = m.hexdigest()
        print(f'Generated features with hash {my_hash}')
    except Exception:
        my_hash = 'unavailable'
    with hdf.File(output_file.name, mode='w') as file:
        # An HDF5 dataset created with the default settings will be contiguous;
        # in other words, laid out on disk in traditional C order. To access
        # all data in a feature a low cost, we store it as
        # (n_features, n_samples).
        if store_hidden:
            file['hidden_features'] = hidden_features.transpose()
            file['hidden_features'].attrs.update({'hidden': True})
            file['hidden_features'].dims[0].label = 'feature'
            file['hidden_features'].dims[1].label = 'sample'
            file['hidden_usefulness'] = hidden_usefulness
            file['hidden_usefulness'].dims[0].label = 'feature'
        file['features'] = out_features.transpose()
        file['features'].attrs.update(generate_args_original_copy)
        file['features'].dims[0].label = 'feature'
        file['features'].dims[1].label = 'sample'
        file['labels'] = out_labels
        file['labels'].dims[0].label = 'sample'
        file['usefulness'] = out_usefulness
        file['usefulness'].dims[0].label = 'feature'
        file['names'] = out_names
        file['names'].dims[0].label = 'feature'
        file['id'] = str(uuid.uuid4())
        file['hash'] = my_hash
        file['created_at'] = datetime.now().strftime(
            "%Y-%m-%dT%H:%M:%S.%f")[:-3]  # len(%f)==6 always
