import glob
import re

import numpy as np
import pandas
import seaborn
from flerken.utils import BaseDict


def findall(pattern, string, alt=None):
    results = re.findall(pattern, string)
    if len(results) == 0:
        results.append(alt)
    return results


def get_finfo(folder: str):
    folder = folder.lower()
    info = {'n_voices': None, 'n_layers': None, 'net_type': None, 'remix': None}

    pattern = '\dsa'  # integer+sa
    results = findall(pattern, folder)
    assert len(results) == 1, 'number of voices specified more than once'
    info['net_type'] = findall('test_(\w+[5-9])', folder)[0]
    info['n_voices'] = results[0]

    results = re.findall('[a-z]*?[5-9]', folder)
    if len(results) == 0:
        info['n_layers'] = None
        # info['net_type'] = None
        info['remix'] = None
    elif len(results) == 1:
        s = results[0]
        info['n_layers'] = findall('\d', s)[0]
        # info['net_type'] = findall('(\w+)r?', s)[0]
        info['remix'] = bool(findall('r', s, False)[0])
    return info


def get_einfo(folder: str):
    folder = folder.lower()
    info = {'lang': None, "subset": None, 'loudness': None}
    info['lang'] = findall('(hindi|spanish|english|others)', folder)[0]
    info['subset'] = findall('(test_unseen|test_seen)', folder)[0]
    info['loudness'] = float(findall('\d[\.\w]*', folder)[0])
    info['gender'] = findall('(mixed|male|female)', folder)[0]
    return info


def get_metrics(path, filter_ext, white_metrics_enabled=False):
    avg_metrics = {'sdr': [], 'sar': [], 'si_sdr': [], 'sir': [], 'isdr': [],'input_sdr':[]}
    configfiles = glob.glob(f'{path}/**/*.json', recursive=True)
    for file_path in configfiles:
        file = BaseDict().load(file_path)
        # Since NaNs are filtered here and they are different for each experiment white metrics are
        # computed here
        if white_metrics_enabled:
            network = findall('(test_\w+[5-9])_\dsa', file_path)
            white_metrics = BaseDict().load(file_path.replace(network[0], 'white_metrics'))
            isdr = white_metrics.get('sdr')
        sdr = file.get('sdr')
        sir = file.get('sir')
        sar = file.get('sar')

        si_sdr = file.get('si-sdr')
        if abs(sdr) > 40 or abs(sir) > 50:
            print(f'{file_path},sdr:{sdr},sir:{sir}')
        avg_metrics['sdr'].append(sdr)
        try:

            assert sir is not None
            assert sar is not None
            assert si_sdr is not None
            if white_metrics_enabled:
                assert isdr is not None
                avg_metrics['isdr'].append(sdr - isdr)
                avg_metrics['input_sdr'].append(isdr)
            avg_metrics['sir'].append(sir)
            avg_metrics['sar'].append(sar)
            avg_metrics['si_sdr'].append(si_sdr)
        except AssertionError:
            print(f'BSS eval metrics from {file_path} are None')
    if white_metrics_enabled:
        avg_metrics['isdr'] = np.array(avg_metrics['isdr'])
        avg_metrics['input_sdr'] = np.array(avg_metrics['input_sdr'])
    avg_metrics['sdr'] = np.array(avg_metrics['sdr'])
    avg_metrics['sir'] = np.array(avg_metrics['sir'])
    avg_metrics['sar'] = np.array(avg_metrics['sar'])
    avg_metrics['si_sdr'] = np.array(avg_metrics['si_sdr'])

    exceptions_nan = (np.isnan(avg_metrics['sdr'])) | (np.isnan(avg_metrics['sir'])) | (np.isnan(avg_metrics['sar']))
    exceptions_inf = (np.isinf(avg_metrics['sdr'])) | (np.isinf(avg_metrics['sir'])) | (np.isinf(avg_metrics['sar']))

    exceptions = exceptions_inf | exceptions_nan

    if white_metrics_enabled:
        exceptions_wm = (np.isinf(avg_metrics['isdr'])) | (np.isnan(avg_metrics['isdr']))
        avg_metrics['isdr'] = avg_metrics['isdr'][~exceptions_wm]
        avg_metrics['input_sdr'] = avg_metrics['input_sdr'][~exceptions_wm]
        avg_metrics['isdr_std'] = np.std(avg_metrics['isdr'])
        avg_metrics['isdr_mean'] = np.mean(avg_metrics['isdr'])

    avg_metrics['sdr'] = avg_metrics['sdr'][~exceptions]
    avg_metrics['sar'] = avg_metrics['sar'][~exceptions]
    avg_metrics['sir'] = avg_metrics['sir'][~exceptions]
    avg_metrics['si_sdr'] = avg_metrics['si_sdr']


    avg_metrics['sdr_std'] = np.std(avg_metrics['sdr'])
    avg_metrics['sar_std'] = np.std(avg_metrics['sar'])
    avg_metrics['sir_std'] = np.std(avg_metrics['sir'])
    avg_metrics['si_sdr_std'] = np.std(avg_metrics['si_sdr'])
    avg_metrics['sdr_mean'] = np.mean(avg_metrics['sdr'])

    avg_metrics['sar_mean'] = np.mean(avg_metrics['sar'])
    avg_metrics['sir_mean'] = np.mean(avg_metrics['sir'])
    avg_metrics['si_sdr_mean'] = np.mean(avg_metrics['si_sdr'])

    avg_metrics['NaN'] = str(np.sum(exceptions_nan.astype(np.uint8)))
    avg_metrics['Inf'] = str(np.sum(exceptions_inf.astype(np.uint8)))
    return avg_metrics


def collect_data(data_paths: dict):
    collection = {}
    for name, path in data_paths.items():
        metrics = get_metrics(path)
        collection[name] = {'sdr': metrics['sdr'], 'sir': metrics['sir'], 'sar': metrics['sar']}
    data_frame_dict = {'name': [], 'sdr': [], 'sir': [], 'sar': [], 'lan': []}
    for name, data in collection.items():
        data_frame_dict['name'].append(name)
        data_frame_dict['sir'].append(data['sir'])
        data_frame_dict['sar'].append(data['sar'])
        data_frame_dict['sdr'].append(data['sdr'])
        data_frame_dict['lan'].append(None)
    return pandas.DataFrame.from_dict(data_frame_dict)


def plot_data(data: pandas.DataFrame, **kwargs):
    seaborn.scatterplot(data=data, **kwargs)


def collet_data_csv(path, skip=0):
    import csv
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lines = []
        for i, row in enumerate(csv_reader):
            if i >= skip:
                print(row)
                lines.append(row)
    return list(zip(*lines))
