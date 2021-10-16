from torch import hann_window

# AUDIO AND VIDEO PARAMETERS
AUDIO_LENGTH = (4 * 2 ** 14 - 1)
N_VIDEO_FRAMES = 100
N_SKELETON_FRAMES = 100

AUDIO_SAMPLERATE = 16384
VIDEO_FRAMERATE = 25
AUDIO_VIDEO_RATE = AUDIO_SAMPLERATE / VIDEO_FRAMERATE

N_FFT = 1022
HOP_LENGTH = 256
N_MEL = 80
STFT_WINDOW = hann_window
SP_FREQ_SHAPE = N_FFT // 2 + 1

VIDEO_FRAMES_IN = 100
VIDEO_FRAMES_OUT = 100

VAL_ITERATIONS = 50
TRAIN_ITERATIONS = 2000

# Kinetics statistics
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

fourier_defaults = {"audio_length": AUDIO_LENGTH,
                    "audio_samplerate": AUDIO_SAMPLERATE,
                    "n_fft": N_FFT,
                    "n_mel": N_MEL,
                    "sp_freq_shape": SP_FREQ_SHAPE,
                    "hop_length": HOP_LENGTH
                    }
test_cfg = {'remix_coef': 1}

dataloader_constructor_defaults = {'audio_key': 'audio',
                                   'video_key': 'frames',
                                   'llcp_key': 'llcp_embed',
                                   "skeleton_key": "landmarks",
                                   'audio_length': AUDIO_LENGTH,
                                   'audio_video_rate': AUDIO_VIDEO_RATE,
                                   'n_video_frames': N_VIDEO_FRAMES,
                                   'n_skeleton_frames': N_SKELETON_FRAMES
                                   }
