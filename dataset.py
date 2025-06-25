import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from g2p_en import G2p
import os

# 1. 定义字符到索引的映射（简化为字符级输入）
def create_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz .!?,"
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0留给padding
    char_to_idx["<PAD>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)

# 2. LJSpeech数据集类
class LJSpeechDataset(Dataset):
    def __init__(self, data_dir, char_to_idx, max_text_len=200, max_mel_len=1000):
        self.data_dir = data_dir
        print(os.path.join(data_dir, "metadata.csv"))
        with open(os.path.join(data_dir, "metadata.csv"), encoding='utf-8') as f:
            lines = f.readlines()

        print("原始行数:", len(lines))
        self.metadata = []
        for line in lines:
            line = line.strip()
            self.metadata.append({"id":line.split('|')[0], "text":line.split('|')[1], "normalized_text":line.split('|')[-1]})
        # self.metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), 
        #                            sep="|", names=["id", "text", "normalized_text"])
        # print(self.metadata.iloc[945]["text"])
        self.char_to_idx = char_to_idx
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
        self.sample_rate = 22050
        self.n_mels = 80
        # self.g2p = G2p()  # 音素转换器（可选）

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 加载文本
        # print(self.metadata[idx])
        text = self.metadata[idx]["normalized_text"].lower()

        # 转为字符序列（或音素）
        text_indices = [self.char_to_idx.get(c, self.char_to_idx["<PAD>"]) for c in text]
        text_indices = text_indices[:self.max_text_len]
        text_indices += [self.char_to_idx["<PAD>"]] * (self.max_text_len - len(text_indices))
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        # 加载音频
        wav_path = os.path.join(self.data_dir, "wavs", f"{self.metadata[idx]['id']}.wav")
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)

        # 计算Mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            n_fft=2048,
            hop_length=256
        )
        mel_spec = torch.tensor(mel_spec).T
        mel_spec = torch.log(mel_spec + 1e-9)  # 对数Mel频谱图

        # 裁剪或填充Mel频谱图
        mel_len = min(mel_spec.size(0), self.max_mel_len)
        mel_spec = mel_spec[:mel_len]
        if mel_len < self.max_mel_len:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 0, 0, self.max_mel_len - mel_len))
        # 停止标志（stop token）
        stop_targets = torch.zeros(self.max_mel_len)
        stop_targets[:mel_len] = 0
        stop_targets[mel_len - 1] = 1  # 最后一帧设为1，表示停止

        return text_tensor, mel_spec, stop_targets, mel_len

# 3. 数据加载器
def get_dataloader(data_dir, batch_size=16):
    char_to_idx, _, vocab_size = create_vocab()
    dataset = LJSpeechDataset(data_dir, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda batch: collate_fn(batch, char_to_idx))
    return dataloader, vocab_size

def collate_fn(batch, char_to_idx):
    texts, mels, stops, mel_lens = zip(*batch)
    texts = torch.stack(texts)
    mels = torch.stack(mels)
    stops = torch.stack(stops)
    mel_lens = torch.tensor(mel_lens, dtype=torch.long)
    return texts, mels, stops, mel_lens

def audio_to_mel(wav_path, sample_rate=22050, n_mels=80, n_fft=2048, hop_length=256):
    audio, sr = librosa.load(wav_path, sr=sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_spectrogram = np.log(mel_spectrogram + 1e-9)
    print(mel_spectrogram)
    mel_spectrogram = torch.tensor(mel_spectrogram.T)  # [n_frames, n_mels]

    return mel_spectrogram

def mel_to_audio(mel_outputs, sample_rate=22050, n_mels=80, n_fft=2048, hop_length=256, win_length=2048):

    mel_outputs = mel_outputs.cpu().numpy()
    mel_outputs = np.exp(mel_outputs)
    mel_outputs = np.maximum(mel_outputs, 1e-9)
    
    # Reconstruct audio from Mel spectrogram using Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel_outputs.T,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    return audio, sample_rate

if __name__ == "__main__":
    data_dir = "/Users/hudaili/Desktop/VsCodeProjects/TTS/data/LJSpeech-1.1"
    wav_path = "/Users/hudaili/Desktop/VsCodeProjects/TTS/data/LJSpeech-1.1/wavs/LJ001-0001.wav"
    # get_dataloader(data_dir)
    char_to_idx, _, vocab_size = create_vocab()
    dataset = LJSpeechDataset(data_dir, char_to_idx)
    for i in range(len(dataset)):
        print(i)
        dataset[i]

    exit(0)
    mel = audio_to_mel(wav_path)
    audio, sr = mel_to_audio(mel)
    torchaudio.save("output.wav", torch.tensor(audio).unsqueeze(0), sr)
    print("Audio saved as output.wav")