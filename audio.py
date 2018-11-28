import numpy as np
import librosa

def SNR(signal, noise):
    A = np.mean(signal ** 2)
    A_noise = np.mean(noise ** 2)
    snr = 20 * np.log10(A / A_noise)
    
    return snr

def snr_coefficient(target_snr, signal, noise):
    A = np.mean(signal ** 2)
    A_noise = np.mean(noise ** 2)
    
    k = 10 ** ((np.log10(A / A_noise) - target_snr / 20) / 2)
    return k


def mix_sig_noise(signal, noise, target_snr):
    k = snr_coefficient(target_snr, signal, noise)
    scaled_noise = k * noise
    
    if len(scaled_noise) <= len(signal):
        n_repeats = len(signal) // len(scaled_noise) + 1
        scaled_noise = np.tile(scaled_noise, n_repeats)
        
    noisy_signal = signal + scaled_noise[:len(signal)]
    noisy_signal[noisy_signal > 1.] = 1.
    noisy_signal[noisy_signal < -1.] = -1.
    return noisy_signal

def get_frames(signal, frame_size, hop_size):
    grid_smp = np.arange(0, signal.shape[0] - frame_size, hop_size)
    frames = np.zeros(shape=(grid_smp.shape[0], frame_size))
    
    for i in range(grid_smp.shape[0]):
        l_i = grid_smp[i]
        r_i = l_i + frame_size
        frames[i, :] = signal[l_i:r_i]
    
    return frames

def get_spectr(frames, fft_size, cut_off, fft_step):
    window = np.hamming(frames.shape[1])
    window = np.reshape(window, (1, frames.shape[1]))
    weighted_frames = frames * window
    
    spectr = np.fft.fft(weighted_frames, n=fft_size, axis=1)
    spectr = spectr / fft_size * 2
    spectr_log = np.log10(abs(spectr)) 
    
    n_env = int(cut_off // fft_step)
    spectr_log = spectr_log[:, :n_env]
    
    return spectr_log

def get_harmonics(spectr, n_cand, n_harm, f0_min, f0_max, fft_step):
    fft_inds = np.zeros(shape=(n_harm, n_cand), dtype=np.int32)
    cands = np.linspace(f0_min, f0_max, n_cand)
    
    for i in range(n_cand):
        indices = np.round((1 + np.arange(0, n_harm)) / 2 * cands[i] / fft_step + 1)
        fft_inds[:, i] = indices.astype(np.int32)

    features = spectr[:, fft_inds]
    features = np.transpose(features, axes=(0, 2, 1))
    return features

def get_labels(signal, hparams):
    frame_size = hparams.frame_size if hparams.frame_size is not None else hparams.fft_size
    frames = get_frames(signal, frame_size, hparams.hop_size)

    frames = np.sum(np.abs(frames), axis=1)
    labels = (frames > hparams.voice_treshold).astype(np.float32)
    
    return labels

def get_features(signal, hparams):
    frame_size = hparams.frame_size if hparams.frame_size is not None else hparams.fft_size
    frames = get_frames(signal, frame_size, hparams.hop_size)
    
    fft_step = hparams.sr / hparams.fft_size
    spectr = get_spectr(frames, hparams.fft_size, hparams.cut_off, fft_step)
    
    features = get_harmonics(spectr, hparams.n_cand, hparams.n_harm, hparams.f0_min, hparams.f0_max, fft_step)
    
    return features.astype(np.float32)