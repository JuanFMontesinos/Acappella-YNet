import streamlit as st
import torch
import numpy as np
import os, subprocess
import cv2
import preproc
from preproc.face_processor import face_processor
from scipy.io.wavfile import read, write
from VnBSS.utils import normalize_max
import imageio
from VnBSS import y_net_gr

from_youtube = st.sidebar.checkbox('Video from youtube')
st.title('Y-net: Audio-Visual Voice separation')
t0 = st.number_input('Initial time', value=124)

duration = int(st.number_input('Track duration (multiple of 4)', value=8))
assert duration % 4 == 0, 'Duration must be a multiple of 4'
n_samples = duration // 4
duration = n_samples * 4
expected_adur = (duration * 2 ** 14 - 1)
expected_vdur = duration * 25

device = st.sidebar.text_input("Device (cpu or cuda:0)", 'cuda:0')
DEVICE = torch.device(device)
MEAN = np.array([0.43216, 0.394666, 0.37645])
STD = np.array([0.22803, 0.22145, 0.216989])

ydl_opts = {
    'format': 'bestvideo+bestaudio',
    'outtmpl': '%(id)s.%(ext)s',
    'cachedir': False,
    """
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    """
    'logger': None
}


def np_int2float(waveform: np.ndarray, raise_error: bool = False) -> np.ndarray:
    """
    Cast an audio array in integer format into float scaling properly .
    :param waveform: numpy array of an audio waveform in int format
    :type waveform: np.ndarray
    :param raise_error: Flag to raise an error if dtype is not int
    """

    if waveform.dtype == np.int8:
        return (waveform / 128).astype(np.float32)

    elif waveform.dtype == np.int16:
        return (waveform / 32768).astype(np.float32)

    elif waveform.dtype == np.int32:
        return (waveform / 2147483648).astype(np.float32)
    elif waveform.dtype == np.int64:
        return (waveform / 9223372036854775808).astype(np.float32)
    elif raise_error:
        raise TypeError(f'int2float input should be of type np.intXX but {waveform.dtype} found')
    else:
        return waveform


@st.cache()
def apply_single(video_path: str, dst_path: str, input_options: list, output_options: list, ext: None):
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output


    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path), f'{video_path} does not exist'
    assert os.path.isdir(os.path.dirname(dst_path))
    if ext is not None:
        dst_path = os.path.splitext(dst_path)[0] + ext
    result = subprocess.Popen(["ffmpeg", *input_options, '-i', video_path, *output_options, dst_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read().decode("utf-8")
    stderr = result.stderr
    if stdout != '':
        print(stdout)
    if stderr is not None:
        print(stderr.read().decode("utf-8"))


@st.cache()
def compute_landmarks(frames):
    MAX_IMAGE_WIDTH = 1024
    mean_face = np.load(os.path.join(preproc.__path__[0], 'mean_face.npy'))
    crop_width, crop_height = face_processor.get_width_height(mean_face)
    fp = face_processor(gpu_id=0, mean_face=mean_face, ref_img=None, img_size=(244, 224))
    processed_frames = []
    processed_landmarks = []
    landmarks = []
    for num_frame in range(frames.shape[0]):
        img_raw = frames[num_frame]
        if img_raw.shape[1] >= MAX_IMAGE_WIDTH:
            asp_ratio = img_raw.shape[0] / img_raw.shape[1]
            dim = (MAX_IMAGE_WIDTH, int(MAX_IMAGE_WIDTH * asp_ratio))
            new_img = cv2.resize(img_raw, dim, interpolation=cv2.INTER_AREA)
            img = np.asarray(new_img)
        else:
            img = img_raw
        warped_img, landmarks, bbox = fp.process_image(img)
        _, aligned_landmarks, _ = fp.process_image(warped_img)
        aligned_landmarks_resized = aligned_landmarks * [96 / crop_width, 128 / crop_height]
        img_frame = warped_img[:crop_height, :crop_width, :]
        img_frame_resized = cv2.resize(img_frame, (96, 128))
        processed_frames.append(img_frame_resized)
        processed_landmarks.append(aligned_landmarks_resized)
    return np.stack(processed_frames), np.stack(processed_landmarks).astype(np.int16)


if from_youtube:
    from youtube_dl import YoutubeDL, DownloadError

    with YoutubeDL(ydl_opts) as ydl:
        raise DownloadError('Mock-up error')  # Comment this line to use your own video from youtube
        ydl.download([video_path])
        from_youtube = True
else:
    import demo_samples

    file_path = st.text_input('File path to video', demo_samples.jacksons_five_full())

input_opts = ['-y', '-ss', str(t0)]
video_opts = ['-t', str(duration + 1), '-r', '25']
audio_opts = ['-t', str(duration + 1), '-ac', '1', '-ar', '16384']
apply_single(file_path, './audio.wav', input_opts, audio_opts, ext='.wav')
apply_single(file_path, './video.wav', input_opts, video_opts, ext='.mp4')
# Displaying the video
st.video('video.mp4', )

audio = np_int2float(read('audio.wav')[1][:expected_adur])  # Reading audio and converting int16 -> float
audio = normalize_max(audio)  # Normalizing wrt abs max
reader = imageio.get_reader('video.mp4')
frames = np.stack([x for x in reader])[:expected_vdur]

# Check the video length is adequate
assert len(audio) == expected_adur, f'Audio length is {len(audio)} but should be {expected_adur}'
assert frames.shape[0] == expected_vdur, f'Video length is {frames.shape[0]} but should be {expected_vdur}'

# Face detection, cropping and alignment
print('Face detection, cropping and alignment ')
print('This may take a while')
cropped_frames, landmarks = compute_landmarks(frames)
imageio.mimwrite('cropped_frames.mp4', [x for x in cropped_frames], fps=25)
inputs = {'src': torch.from_numpy(audio).unsqueeze(0).to(DEVICE)}
st.video('cropped_frames.mp4')

model = st.sidebar.selectbox('Select the model', ['Y-net-mr', 'Y-net-gr'])

# Requiered landmark's shape: N,C,T,V,M
# Permute from TVC->CTV
# Unsqueeze to generate N and M
inputs['landmarks'] = torch.from_numpy(landmarks).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).float().to(DEVICE)
model = y_net_gr(n=n_samples).to(DEVICE)
model._window = model._window.to(DEVICE)
model.eval()
with torch.no_grad():
    pred = model(inputs, real_sample=True)
estimated_sp = pred['estimated_sp'][0].cpu().numpy()
estimated_wav = pred['estimated_wav'][0].cpu().numpy()
write('estimated_wav.wav', 16384, estimated_wav)
st.audio('estimated_wav.wav', format="audio/wav", start_time=0)
