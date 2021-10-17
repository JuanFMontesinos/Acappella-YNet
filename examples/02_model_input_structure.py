import torch

import VnBSS

if __name__ == '__main__':
    """
    audio: the main audio waveform of shape N,M
    audio_acmt: the secondary audio waveform of shame N,M
    src: If using inference on real mixtures, the mixture audio waveform of shape N,M
    sk: skeletons of shape N,C,T,V,M which is batch, channels, temporal, joints, n_people
    """
    SAMPLING_RATE = 16384  # Audio sampling rate
    FRAME_RATE = 25  # Video Framerate
    duration = 4

    model = VnBSS.y_net_gr()
    # For Y-Net-gr we need audio and landmarks
    # Inference with a real sample, we need to pass 'src'
    with torch.no_grad():
        model.eval()
        landmarks = torch.rand(1, 2, duration * FRAME_RATE, 68, 1)
        waveform = torch.rand(1, duration * SAMPLING_RATE - 1)
        inputs = {'src': waveform, 'landmarks': landmarks}
        print(f'Waveform shape: {waveform.shape}')
        pred = model(inputs, real_sample=True)
        print(pred.keys())

    # Training with artificial mix
    model.train()
    landmarks = torch.rand(1, 2, duration * FRAME_RATE, 68, 1)
    target_speaker_waveform = torch.rand(1, duration * SAMPLING_RATE - 1)
    accompaniment_audio = torch.rand(1, duration * SAMPLING_RATE - 1)
    inputs = {'audio': target_speaker_waveform, 'audio_acmt': accompaniment_audio, 'landmarks': landmarks}
    model(inputs, real_sample=False)
    # Here estimated waveform is not computed to save resources
    # model.training triggers the iSTFT computation
