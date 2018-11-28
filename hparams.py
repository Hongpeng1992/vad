import tensorflow as tf

hparams = tf.contrib.training.HParams(
    n_harm = 14, # количество гармоник
    n_cand = 100, # количество кандидатов для F0
    n_hidden = 16, # количество нейронов в скрытом слое
    
    sr = 16000, # частота дискретизации сигнала
    snr_rates = [-10, -5, 0, 5, 10, 15, 20], # значения SNR для аугументации речевых записей шумами
    
    cut_off = 4000, # максимальная частота сигнала
    fft_size = 1024, # размер быстрого преобразования Фурье
    frame_size = 800, # размер анализируемого фрейма, если None, то frame_fize=fft_size
    hop_size = 200, # размер смещения фреймов
    
    f0_min = 70, # минимальное значение частоты основного тона
    f0_max = 350, # максимальное значение частоты основного тона
    voice_treshold = 3, # 2
    n_valid_files = 20,
    
    batch_size = 128,
    learning_rate = 0.01,
    momentum = 0.7,
    shuffle_buffer_size = 1024,
    random_seed = 42,
    
    #параметры нормировки признаков
    normalize_features = True,
    mean_std_path = 'mean-std.npz',
    
    eps = 1e-7
)