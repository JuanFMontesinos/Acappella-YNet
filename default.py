import argparse

from flerken.utils import BaseDict

__all__ = ['DEBUG', 'DUMP_FILES', 'SEND', 'argparse_default']

POSSIBLE_DATASETS = ['test_unseen_english_female', 'test_unseen_english_male',
                     'test_unseen_hindi_female', 'test_unseen_hindi_male', 'test_unseen_others_female',
                     'test_unseen_others_male', 'test_unseen_spanish_male', 'test_unseen_spanish_female']

DEBUG = BaseDict({'isnan': True, 'ds_autogen': False, "overfit": False, 'verbose': False})

DUMP_FILES = {'enabled': True,
              'force': False,  # Skips all the conditions,
              'audio': False,
              'landmarks': False,
              'masks': False,
              'video': False,
              'train': {'enabled': False, 'iter_freq': 1000, 'epoch_freq': 4},
              'val_seen': {'enabled': False, 'iter_freq': 50, 'epoch_freq': 4},
              }

SEND = {'bss_eval': ('bss',),
        'loss': ('pred',),
        'acc': ('pred', 'gt')}

DEFAULT_TRACE_PATH = './traces'


def argparse_default():
    parser = argparse.ArgumentParser(description='Y-Net training')
    parser.add_argument('-m', '--model', help='model name', type=str, required=True)

    parser.add_argument('--workname', help='Experiment name', type=str, default=None)
    parser.add_argument('--arxiv_path', help='Main directory for all the experiments',
                        type=str, default='./debug_dir')
    parser.add_argument('--pretrained_from', help='Use some weights to start from',
                        type=str, default=None)

    parser.add_argument('--device', help='Device for training the experiments',
                        type=str, default='cuda:0')

    parser.add_argument('--force_dumping', dest='dumping', action='store_true')
    parser.add_argument('--legacy', dest='legacy', action='store_true')
    parser.add_argument('--white_metrics', dest='white_metrics', action='store_true')
    parser.add_argument('--testing', dest='testing', action='store_true')
    parser.set_defaults(testing=False, dumping=False, remix=False)
    parser.add_argument('--test_in', nargs='+', type=str, default=[],
                        help=f'List of datasets to test on,allowed ones {POSSIBLE_DATASETS}')
    parser.add_argument('--remix', dest='remix', action='store_true',
                        help='Set remix true or false, this is only used in testing')
    parser.add_argument('--loudness_levels', nargs='+', type=float, default=[1.], help='loudness levels for evaluation')
    parser.add_argument('--loudness_train', default=1, help='loudness levels')
    args= parser.parse_args()
    try:
        args.loudness_train = float(args.loudness_train)
    except ValueError:
        pass
    return args
