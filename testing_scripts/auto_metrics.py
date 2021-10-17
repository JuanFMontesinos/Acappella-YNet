import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from scipy.stats import gaussian_kde

from VnBSS.utils.metrics import get_metrics, get_finfo, get_einfo


def plot_2sa_unseen_by_model(summary):
    sns.set_theme(style="darkgrid")
    subset = summary[summary['n_voices'] == '2sa']
    subset = subset[subset['subset'] == 'test_unseen']
    subset = subset[subset['lang'].isnull()]  # No language, thus, test_unseen_mixed
    Z_MAX = 30
    for (idx, model_row) in subset.iterrows():
        model = model_row.net_type
        model = model.replace('5', '').replace('7', '').replace('_', '-').replace('y', 'Y').replace('net', 'Net')
        model = model.replace('llcp', 'LLCP')
        isdr = model_row.isdr
        input_sdr = model_row.input_sdr
        xy = np.vstack([input_sdr, isdr])
        z = gaussian_kde(xy)(xy)
        plt.figure()
        sns.scatterplot(x=input_sdr, y=isdr, c=z, cmap='coolwarm')
        # sns.histplot(x=input_sdr, y=isdr, bins=30, pthresh=.1, cmap='rocket_r')
        # sns.kdeplot(x=input_sdr, y=isdr, levels=5, color="rosybrown", linewidths=1)
        plt.axhline(y=0, xmin=0, xmax=1, linestyle='dashed', label='0 dBs line', color='fuchsia')
        plt.title(f'{model}')
        plt.xlabel('Input SDR (dB)')
        plt.ylabel('SDR improvements (dB)')
        plt.text(-16, 0.4, '0 dBs line', color='fuchsia')
        plt.show()
    return None


if __name__ == '__main__':
    exp_path = '/media/jfm/SlaveEVO970/AcapellaEx'
    dst = '/home/jfm'
    white_metrics = False
    summary = {}

    exp_folders = [x for x in os.listdir(exp_path) if 'test' in x]
    exp_fpaths = [os.path.join(exp_path, x) for x in exp_folders]
    for path in exp_fpaths:

        finfo = get_finfo(os.path.basename(path))
        test_folders = [x for x in os.listdir(path) if 'test' in x]
        metric_els = ['sdr', 'sir', 'sar', 'si_sdr', 'isdr', 'NaN', 'Inf',
                      'sdr_mean', 'sdr_std',
                      'sir_mean', 'sir_std',
                      'sar_mean', 'sar_std', ]
        if white_metrics:
            metric_els = metric_els + ['input_sdr', 'isdr_mean', 'isdr_std', ]
        info_els = ['lang', 'subset', 'loudness', 'gender']
        metrics = {x: [] for x in metric_els}
        einfo = {x: [] for x in info_els}
        for results in [get_einfo(x) for x in test_folders]:
            for el in info_els:
                einfo[el].append(results[el])
        for results in [get_metrics(os.path.join(path, x), white_metrics_enabled=white_metrics, filter_ext=True) for x
                        in
                        test_folders]:
            for el in metric_els:
                metrics[el].append(results[el])

        finfo = {name: [val for _ in range(len(metrics['sdr']))] for name, val in finfo.items()}
        finfo.update(metrics)
        finfo.update(einfo)
        if not bool(summary):  # if dictionary is empty
            summary.update(finfo)
        else:
            for key in summary:
                summary[key].extend(finfo[key])

    summary = pandas.DataFrame.from_dict(summary).sort_values(by=['net_type', 'n_voices', 'subset', 'lang'])
    # TO SAVE THE FILE AS CSV
    # summary.to_csv(os.path.join(dst,f'metrics_{str(datetime.datetime.now())[:-7]}.csv'),index=False,header=True)
    # summary.to_csv(os.path.join(dst,f'metrics.csv'),index=False,header=True)
    if white_metrics:
        plot_2sa_unseen_by_model(summary)

    for i in summary.iterrows():
        print(i)
        input()

    print(summary)
