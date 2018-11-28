import os
import numpy as np
from tqdm import tqdm
from hparams import hparams
from audio import get_features, get_labels, mix_sig_noise, snr_coefficient
import argparse
import multiprocessing
import librosa

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def get_wav_files(path):
    filenames = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.wav'):
            filenames.append(os.path.join(path, f))
            
    return filenames


def process_file(speech_filename, hparams, noisy_filenames, target_folder):
    signal, _ = librosa.load(speech_filename, sr=hparams.sr)
    
    noise_filename = np.random.choice(noisy_filenames)
    noise, _ = librosa.load(noise_filename, sr=hparams.sr)
    
    target_snr = np.random.choice(hparams.snr_rates)
    noisy_signal = mix_sig_noise(signal, noise, target_snr)
    
    features = get_features(noisy_signal, hparams)
    labels = get_labels(signal, hparams)
        
    target_filename = os.path.join(target_folder, os.path.basename(speech_filename))
    np.savez(target_filename, features=features, labels=labels)
    
    k = snr_coefficient(target_snr, signal, noise)
    scaled_noise = k * noise
    noise_features = get_features(scaled_noise, hparams)
    noise_labels = np.zeros(shape=(noise_features.shape[0]), dtype=np.float32)
    target_filename = os.path.join(target_folder, os.path.basename(noise_filename))
    np.savez(target_filename, features=noise_features, labels=noise_labels)
    
    return len(signal)




if __name__ == '__main__':
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--n_jobs', type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    
    SPEECH_DATA_PATH = os.path.join(args.base_dir, 'speech/librivox')
    NOISE1_DATA_PATH = os.path.join(args.base_dir, 'noise/sound-bible')
    NOISE2_DATA_PATH = os.path.join(args.base_dir, 'noise/free-sound')
    
    MUSIC_DATA_PATH = os.path.join(args.base_dir, 'music')
    
    MUSIC_FOLDERS = ['fma', 'fma-western-art', 'hd-classical', 'rfm']
    TARGET_FOLDER = args.output
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    
    speech_filenames = get_wav_files(SPEECH_DATA_PATH)

    noisy_filenames = get_wav_files(NOISE1_DATA_PATH)
    noisy_filenames += get_wav_files(NOISE2_DATA_PATH)
    
    executor = ProcessPoolExecutor(max_workers=args.n_jobs)
    futures = []
    index = 1
    for speech_filename in speech_filenames:
        futures.append(executor.submit(partial(process_file, speech_filename,  hparams, noisy_filenames, TARGET_FOLDER)))
        index += 1

    singal_lengths =  [future.result() for future in tqdm(futures) if future.result() is not None]
    signal_length = sum(singal_lengths) / hparams.sr / 60 / 60
    
    print(f'Computing meand and std...', end='')
    filenames = [f for f in os.listdir(TARGET_FOLDER) if f.endswith('.npz')][:100]
    
    data = [np.load(os.path.join(TARGET_FOLDER, f))['features'] for f in filenames]
    data = np.concatenate(data, axis=0)
    
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    np.savez(hparams.mean_std_path, mean=mean, std=std)
    print('[OK]')
    
    print(f'Total samples: {signal_length:.2f} hours')
   