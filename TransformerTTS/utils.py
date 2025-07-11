import numpy as np
import librosa
import os, copy
from scipy import signal
import hyperparams as hp
import torch as t

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    # mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel_basis = librosa.filters.mel(
        sr=hp.sr,
        n_fft=hp.n_fft,
        n_mels=hp.n_mels
    )
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        # est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        est = librosa.stft(
            y=X_t,
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length
        )
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(
        spectrogram,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window="hann"
    )
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return t.from_numpy(position_enc).type(t.FloatTensor)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return t.FloatTensor(sinusoid_table)

def guided_attention(N, T, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W

import numpy as np
import librosa
import soundfile as sf

def mel_to_audio(mel_spectrogram, n_iter=60, output_wav_path=None):
    """
    将 mel 频谱转换回音频波形（使用 Griffin-Lim 算法）
    
    参数:
        mel_spectrogram (np.ndarray): 形状为 (T, n_mels) 的 mel 频谱
        hp: 超参数对象，必须包含以下属性：
            - ref_db
            - max_db
            - power
            - sr (采样率)
            - n_fft
            - hop_length
            - win_length
            - fmin
            - fmax
        n_iter (int): Griffin-Lim 迭代次数，默认 60
        output_wav_path (str or None): 如果不为 None，则保存音频文件到该路径
    
    返回:
        wav (np.ndarray): 合成的音频波形
    """
    # Step 1: 反归一化 mel 频谱
    print("YYYYYYY")
    mel = mel_spectrogram.copy()
    mel = np.clip(mel, 1e-8, 1)
    mel = mel * hp.max_db - hp.max_db + hp.ref_db  # 反标准化
    mel = 10 ** (mel / 20)  # 转回线性幅度域

    # Step 2: 构建 mel 滤波器组并反变换到 magnitude spectrogram
    mel_basis = librosa.filters.mel(
        sr=hp.sr,
        n_fft=hp.n_fft,
        n_mels=hp.n_mels
    )

    # 伪逆矩阵（Pseudo-inverse）
    inv_mel_basis = np.linalg.pinv(mel_basis)

    # 使用伪逆恢复 magnitude spectrogram
    mag = np.dot(inv_mel_basis, mel.T)
    mag = np.maximum(1e-10, mag)  # 防止零值

    # Step 3: 使用 Griffin-Lim 算法估计相位，重构音频
    spec = mag.astype(np.complex64) * np.exp(np.angle(librosa.stft(np.random.randn(256), 
                                                                  n_fft=hp.n_fft, 
                                                                  hop_length=hp.hop_length))[:, :mag.shape[1]])

    wav = librosa.istft(spec, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")

    for _ in range(n_iter):
        spec = librosa.stft(wav, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
        _, phase = librosa.magphase(spec)
        spec_new = mag * phase
        wav = librosa.istft(spec_new, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")

    # Step 4: 去预加重
    wav = np.append(wav[0], wav[1:] + hp.preemphasis * wav[:-1])

    # Step 5: 限制音量范围
    wav = np.clip(wav, -1, 1)

    # Step 6: 保存音频（可选）
    if output_wav_path is not None:
        sf.write(output_wav_path, wav, hp.sr)

    return wav