import numpy as np
import librosa
import librosa.display
import os


def get_frames(signal, frame_size, frame_offset):
    grid_smp = np.arange(0, signal.shape[0] - frame_size, frame_offset)
    frames = np.zeros(shape=(grid_smp.shape[0], frame_size))
    
    for i in range(grid_smp.shape[0]):
        l_i = grid_smp[i]
        r_i = l_i + frame_size
        frames[i, :] = signal[l_i:r_i]
    
    return frames


def get_spectr(frames, fft_size, fft_step, cut_off):
    window = np.hamming(frames.shape[1])
    window = np.reshape(window, (1, frames.shape[1]))
    weighted_frames = frames * window
    
    spectr = np.fft.fft(weighted_frames, n=fft_size, axis=1)
    spectr = spectr / fft_size * 2
    spectr_log = np.log10(abs(spectr)) 
    
    n_env = int(cut_off // fft_step)
    spectr_log = spectr_log[:, :n_env]
    
    return spectr_log


def get_harmonics(spectr, n_harm, n_cand, f0_range, fft_step):
    fft_inds = np.zeros(shape=(n_harm, n_cand), dtype=np.int32)
    cands = np.linspace(f0_range[0], f0_range[1], n_cand)
    
    for i in range(n_cand):
        indices = np.round((1 + np.arange(0, n_harm)) / 2 * cands[i] / fft_step + 1)
        fft_inds[:, i] = indices.astype(np.int32)

    features = spectr[:, fft_inds]
    features = np.transpose(features, axes=(0, 2, 1))
#     features = features[..., np.newaxis]

    return features


def get_features(signal=None, filename=None, sr=16000, 
                 fft_size=512, frame_size=512, frame_offset=128, cut_off=4000, 
                 n_harm=14, n_cand=100, f0_range=(70, 350)):
    
    fft_step = sr / fft_size
    if signal is None:
        signal, _ = librosa.load(filename, sr=sr)
        
    frames = get_frames(signal, frame_size, frame_offset)
    spectr = get_spectr(frames, fft_size, fft_step, cut_off)
    features = get_harmonics(spectr, n_harm, n_cand, f0_range, fft_step)
    
    return features.astype(np.float32)


