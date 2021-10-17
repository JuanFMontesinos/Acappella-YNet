import streamlit as st
import os
from flerken.utils import BaseDict
import glob
import numpy as np

from VnBSS.utils.metrics import get_metrics as _gm


@st.cache()
def get_metrics(path):
    return _gm(path)  # wrapper around this function to use st.cache


def return_files(path):
    files = sorted([x for x in os.listdir(path)])
    return files, len(files)


def get_audio_files(path):
    files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(('.mp3', '.wav'))])
    yield from files


def get_video_files(path):
    files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.mp4')])
    yield from files


def get_images_files(path):
    files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.png')])
    yield from files


def get_metadata(path):
    files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.json')])
    for p in files:
        yield BaseDict().load(p), p


json_files = {}
st.title('Training and eval visualization')
root = st.text_input('Folder to visualization default_path')
# if os.default_path.exists(root) and root != '':
#     json_files = get_metrics(root)
#     json_files
# Get all the elements (independent samples) predicted
path_list, slicer_nmax = return_files(root)
# Choose one with the slider
idx = st.sidebar.slider('Select a file', 0, slicer_nmax)
filename = path_list[idx]
st.sidebar.text(filename)

# Get all the info about that element
path = os.path.join(root, filename)
for f, p in get_metadata(path):
    st.text(f'File {p}')
    st.json(f)
for audio_path in get_audio_files(path):
    st.text(f'File: {audio_path}')
    st.audio(audio_path)

for video_path in get_video_files(path):
    st.text(f'File: {video_path}')
    st.video(video_path)

for image_path in get_images_files(path):
    st.text(f'File: {image_path}')
    st.image(image_path, format='PNG')
