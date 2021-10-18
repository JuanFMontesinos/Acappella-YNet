import os

import torch
from flerken.framework.framework import Model
from flerken.utils import BaseDict
from google_drive_downloader import GoogleDriveDownloader as ggd

COMPUTE = True
BATCH_SIZE = 10
MODEL = 'y_net_gr'
n = 1

DATA_PATH = './the_circle_of_life'

# ===================================================================================================
# ===================================================================================================
if not os.path.exists('./the_circle_of_life'):
    ggd.download_file_from_google_drive('1An3kalwUpyPWpeH_urJchWsWaffVj3_J', './the_circle_of_life.zip', unzip=True)
    os.remove('./the_circle_of_life.zip')
DEBUG = BaseDict({'verbose': False, 'isnan': True, 'ds_autogen': False, 'audio_length': True,
                  'video_example': False})
constructor = getattr(__import__('VnBSS'), 'y_net_gr')
model = constructor(pretrained=True, n=n, debug_dict=DEBUG)
DST = os.path.join(DATA_PATH, 'frames_npy', 'estimated_samples', MODEL)
DEVICE = torch.device('cuda:0')

# CONSTANTS
ROLES = ['antilope1', 'antilope2', 'backtrack', 'leonas', 'leones', 'rafiki']
SR = 16384
FPS = 25
TEMPORAL_FEATS = 16  # n temporal fetures expected in the bottleneck per 4s of track
AUDIO_SAMPLE = 65535  # Audio length per 4s
CROP = 34  # N seconds to crop from the beginning

# PATHS
frames_path = os.path.join(DATA_PATH, 'frames')
audios_path = os.path.join(DATA_PATH, 'audio')
embeddings_path = os.path.join(DATA_PATH, 'llcp_embed')
landmarks_path = os.path.join(DATA_PATH, 'landmarks')
original_vd = os.path.join(DATA_PATH, 'samples')

gt_ad_path = os.path.join(audios_path, 'the_circle_of_life.wav')
paths = {}
for role in ROLES:
    paths[f'{role}_vd_path'] = os.path.join(frames_path, f'{role}.npy')
    paths[f'{role}_or_path'] = os.path.join(original_vd, f'{role}.mp4')
    paths[f'{role}_ad_path'] = os.path.join(audios_path, f'{role}.wav')
    paths[f'{role}_em_path'] = os.path.join(embeddings_path, f'{role}.npy')
    paths[f'{role}_ld_path'] = os.path.join(landmarks_path, f'{role}.npy')

if COMPUTE:
    import numpy as np
    from scipy.io.wavfile import read, write
    from flerken.audio import np_int2float
    import config
    from torch_mir_eval import bss_eval_sources

    cfg_path = config.__path__[0]
    torch.set_grad_enabled(False)

    # Crop ground-truth
    gt_ad = torch.from_numpy(np_int2float(read(gt_ad_path)[1])[CROP * SR:]).to(DEVICE)

    for role in ROLES:
        # plt.imshow(np.load(paths[f'{role}_vd_path'])[CROP * 4 * FPS:][100])
        # plt.show()
        globals()[f'{role}_vd'] = torch.from_numpy(
            np.load(paths[f'{role}_vd_path'])[CROP * FPS:]).to(DEVICE)
        sr, globals()[f'{role}_ad'] = read(paths[f'{role}_ad_path'])
        globals()[f'{role}_ad'] = torch.from_numpy(
            np_int2float(globals()[f'{role}_ad'])[CROP * SR:]).to(DEVICE)
        globals()[f'{role}_est'] = torch.zeros_like(globals()[f'{role}_ad'])
        assert sr == SR, f'Sampling rate must be {SR} but {sr} found'
        globals()[f'{role}_em'] = torch.from_numpy(
            np.load(paths[f'{role}_em_path'])[CROP * FPS:]).to(DEVICE)
        globals()[f'{role}_ld'] = torch.from_numpy(
            np.load(paths[f'{role}_ld_path'])[CROP * FPS:]).float().to(DEVICE)
    if not os.path.exists(DST):
        os.makedirs(DST)

    # BACKWARD COMPATIBILITY MODIFICATION

    model.eval()
    model = Model(model)
    model.to(DEVICE)
    mean = torch.tensor(config.MEAN).to(DEVICE)
    std = torch.tensor(config.STD).to(DEVICE)
    # INFERENCE TIME
    rvlen = rafiki_vd.shape[0]
    n_chunks = rvlen // (4 * FPS * n)
    cropped_mix = gt_ad[:n_chunks * AUDIO_SAMPLE * n].view(n_chunks, AUDIO_SAMPLE * n)
    # plt.plot(gt_ad.cpu().numpy())
    # plt.show()
    for role in ROLES:
        video = globals()[f'{role}_vd'][:n_chunks * 4 * FPS * n]
        video = (video / 255. - mean) / std
        # Crop around the mouth
        video = video[..., 60:100, 20:80, :]
        # plt.imshow((video[100]*std+mean).cpu().numpy())
        # plt.show()
        video = video.permute(0, 3, 1, 2)

        video = torch.nn.functional.upsample_bilinear(
            video, size=(96, 96))
        video = video.view(n_chunks, 4 * FPS * n, 3, 96, 96).permute(0, 2, 1, 3, 4)

        embedding = globals()[f'{role}_em'][:n_chunks * 4 * FPS * n].view(n_chunks, 4 * FPS * n, 512)
        landmarks = globals()[f'{role}_ld'][:n_chunks * 4 * FPS * n].view(n_chunks, 4 * FPS * n, 68, 2).permute(0, 3, 1,
                                                                                                                2).unsqueeze(
            -1)
        # (N, in_channels, T_{in}, V_{in}, M_{in})
        c = BATCH_SIZE
        for i in range(n_chunks // c):
            inputs = {'src': cropped_mix[c * i:c * (i + 1)],
                      'llcp_embedding': embedding[c * i:c * (i + 1)],
                      'video': video[c * i:c * (i + 1)],
                      'landmarks': landmarks[c * i:c * (i + 1)]}
            predictions = model.model.forward(inputs, real_sample=True)
            est_audio = predictions['estimated_wav'].reshape(-1)
            globals()[f'{role}_est'][AUDIO_SAMPLE * c * i:c * AUDIO_SAMPLE * (i + 1)] = est_audio
    for role in ROLES:
        sample = globals()[f'{role}_est'][:n_chunks * AUDIO_SAMPLE * n]
        write(os.path.join(DST, f'{role}_est.wav'), SR, (sample / sample.abs().max()).cpu().numpy())
    estimated = torch.stack([globals()[f'{role}_est'][:n_chunks * AUDIO_SAMPLE * n] for role in ROLES])
    gt = torch.stack([globals()[f'{role}_ad'][:n_chunks * AUDIO_SAMPLE * n] for role in ROLES])
    metrics = bss_eval_sources(gt.unsqueeze(0), estimated.unsqueeze(0))
    # metrics = mir(gt.cpu().numpy(), estimated.cpu().numpy(), compute_permutation=False)
    print(metrics)
else:
    import streamlit as st

    for role in ROLES:
        st.subheader(f'{role}')
        st.video(paths[f'{role}_or_path'])
        st.write(f'Ground Truth')
        # st.audio(globals()[f'{role}_ad_path'])
        st.write(f'Estimated')
        st.audio(os.path.join(DST, f'{role}_est.wav'))
