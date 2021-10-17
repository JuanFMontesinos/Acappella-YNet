import csv
import json
import os

import streamlit as st
import youtube_dl
from tqdm import tqdm

from flerken.utils import BaseDict
from flerken.video.utils import apply_single


class YouTubeSaver(object):
    """Load video from YouTube using an auditionDataset.json """

    def __init__(self):
        self.outtmpl = '%(id)s.%(ext)s'
        self.ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': self.outtmpl,

            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'logger': None
        }

    def from_json(self, json_path, dataset_dir, tag2text, ids_info):
        dataset = json.load(open(json_path))
        remaining = BaseDict().load(json_path)
        for tag in dataset.keys():
            text = tag2text[tag]
            # self.ydl_opts['outtmpl'] = os.path.join(dataset_dir, text, self.outtmpl)

            if not os.path.exists(os.path.join(dataset_dir, text)):
                os.makedirs(os.path.join(dataset_dir, text))
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                for video_id in dataset[tag]:
                    try:
                        info = ids_info[video_id]
                        if info['end_seconds'] - info['start_seconds'] < MIN_DUR:
                            continue
                        src = f'{video_id}.wav'
                        dst = os.path.join(dataset_dir, text, src)
                        ydl.download(['https://www.youtube.com/watch?v=%s' % video_id])
                        remaining[tag].pop(remaining[tag].index(video_id))
                        apply_single(src, dst,
                                     input_options=['-ss', str(int(info['start_seconds']))],
                                     output_options=['-to', str(int(info['end_seconds'])),
                                                     '-ac', '1',
                                                     '-ar', str(AUDIO_SR)],
                                     ext='.wav'
                                     )
                        os.remove(src)
                    except KeyError as ex:
                        raise ex
                    except FileNotFoundError as ex:
                        raise ex
                    except Exception:
                        remaining.write(json_path)


def read_csv(path: str):
    with open(path, mode='r') as file:
        csvfile = csv.reader(file)
        yield from csvfile


def read_audioset_csv(path: str):
    print(f'Subset: {path}')
    iterable = read_csv(path)
    print(next(iterable))
    info = next(iterable)
    print(info)
    N = int(info[0].split('=')[1])
    header = next(iterable)
    for line in tqdm(iterable, total=N):
        prop = dict()
        prop[header[0]] = line[0]
        prop['start_seconds'] = float(line[1])
        prop['end_seconds'] = float(line[2])
        prop['positive_labels'] = [x.replace(' ', '').replace('"', '') for x in line[3:]]
        yield prop


def csv2json(path: str):
    ids_info = BaseDict()
    category_info = BaseDict()
    for sample in read_audioset_csv(path):
        for cat in sample['positive_labels']:
            if cat not in category_info:
                category_info[cat] = []
            category_info[cat].append(sample['# YTID'])
        ID = sample['# YTID']
        sample.pop('# YTID')
        ids_info[ID] = sample
    return ids_info, category_info


def extract_json_files():
    ids_info_utrain, cat_info_utrain = csv2json('./unbalanced_train_segments.csv')
    ids_info_utrain.write('./unbalanced_train_ids_info.json')
    cat_info_utrain.write('./unbalanced_train_cat_info.json')

    ids_info_train, cat_info_train = csv2json('./balanced_train_segments.csv')
    ids_info_train.write('./balanced_train_ids_info.json')
    cat_info_train.write('./balanced_train_cat_info.json')

    ids_info_eval, cat_info_eval = csv2json('./eval_segments.csv')
    ids_info_eval.write('./eval_ids_info.json')
    cat_info_eval.write('./eval_cat_info.json')
    return ids_info_utrain, ids_info_train, ids_info_eval, \
           cat_info_utrain, cat_info_train, cat_info_eval


def load_cat_json_file(tag_options, path):
    cat_info = BaseDict().load(path)
    for key in list(cat_info.keys()):
        if key not in tag_options:
            cat_info.pop(key)
    return cat_info


@st.cache()
def load_ontology(path='./ontology.json'):
    with open(path, 'r') as f:
        datastore = json.load(f)
        return datastore


@st.cache()
def translators(ontology):
    tag2text = {}
    text2tag = {}
    for el in ontology:
        name = el['name']
        tag = el['id']
        text2tag[name] = tag
        tag2text[tag] = name
    return tag2text, text2tag


reset_backup = False
reset_info = False
ontology = load_ontology()
tag2text, text2tag = translators(ontology)

st.text('Ontology')
st.json(ontology)

human_options = st.multiselect(
    'Which classes would you like to download',
    list(text2tag.keys()), default=['Beatboxing', 'Whistling', 'A capella', 'Theremin', 'Choir', 'Rapping', 'Yodeling'])
trainin_sets = st.multiselect(
    'Which subsets would you like to download',
    ['unbalanced_train', 'balanced_train', 'eval'])
tag_options = [text2tag[x] for x in human_options]
if not os.path.exists('./unbalanced_train_ids_info.json'):
    extract_json_files()
downloader = YouTubeSaver()
dst = st.text_input('Destiny path')
st.text(f'Path {dst} doesnt exist')

MIN_DUR = st.number_input('Minimum duration', min_value=0, value=8)
AUDIO_SR = st.number_input('Audio sampling rate', min_value=0, value=16384)

reset_backup = st.sidebar.button('Reset backup files')
reset_info = st.sidebar.button('Reset info files')
if reset_backup:
    for path in trainin_sets:
        os.remove(f'./backup{path}_cat_info.json')
if reset_info:
    for path in trainin_sets:
        os.remove(f'./{path}_ids_info.json')
        os.remove(f'./{path}_cat_info.json')
run_key = st.button('Click to run', key=None)
if os.path.exists(dst) and run_key:
    for path in trainin_sets:
        ids_info = BaseDict().load(f'./{path}_ids_info.json')
        cat_info = load_cat_json_file(tag_options, f'./{path}_cat_info.json')
        if not os.path.exists(f'./backup{path}_cat_info.json'):
            cat_info.write(f'./backup{path}_cat_info.json')
        dst_i = os.path.join(dst, path)
        downloader.from_json(f'./backup{path}_cat_info.json', dst_i, tag2text, ids_info)
