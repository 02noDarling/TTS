import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from g2p_en import G2p
import os
from utils import *
from text import text_to_sequence

# 1. 定义字符到索引的映射（简化为字符级输入）
def create_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz .!?,"
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0留给padding
    char_to_idx["<PAD>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)

# class LJSpeechDataset(Dataset):
#     def __init__(self, data_dir, char_to_idx, max_text_len=200, max_mel_len=1000):
#         self.data_dir = data_dir
#         self.char_to_idx = char_to_idx
#         self.max_text_len = max_text_len
#         self.max_mel_len = max_mel_len
#         self.sample_rate = 22050
#         self.n_mels = 80

#         # 存储所有样本的数据
#         self.data = []

#         stats = torch.load('mel_stats.pt')
#         self.mel_mean = stats['mean']
#         self.mel_std = stats['std']


#         metadata_path = os.path.join(data_dir, "metadata.csv")
#         print(f"Reading and preprocessing dataset...")

#         with open(metadata_path, encoding='utf-8') as f:
#             lines = f.readlines()

#         for line in lines:
#             parts = line.strip().split('|')
#             wav_id = parts[0]
#             wav_path = os.path.join(data_dir, "wavs", f"{wav_id}.wav")

#             if not os.path.exists(wav_path):
#                 continue

#             text = parts[-1].lower()  # 使用 normalized_text

#             # 文本编码
#             text_indices = [char_to_idx.get(c, char_to_idx["<PAD>"]) for c in text]
#             text_tensor = torch.tensor(text_indices, dtype=torch.long)

#             # 音频加载和 Mel 谱图计算
#             audio, sr = librosa.load(wav_path, sr=self.sample_rate)
#             mel_spec = librosa.feature.melspectrogram(
#                 y=audio,
#                 sr=sr,
#                 n_mels=self.n_mels,
#                 n_fft=1024,
#                 hop_length=256,
#                 win_length=1024,
#                 fmin=0,
#                 fmax=8000,
#                 power=2.0
#             )
#             mel_spec = torch.tensor(mel_spec).T
#             mel_spec = torch.log(mel_spec + 1e-9)  # 对数压缩
#             mel_spec = (mel_spec - self.mel_mean) / self.mel_std

#             # 停止标志
#             stop_targets = torch.zeros(mel_spec.size(0))
#             stop_targets[-1] = 1  # 最后一帧为停止标记

#             # 添加到数据列表
#             self.data.append((
#                 text_tensor,
#                 mel_spec,
#                 stop_targets
#             ))

#         print(f"Loaded {len(self.data)} samples into memory.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# 原本的dataset
# class LJSpeechDataset(Dataset):
#     def __init__(self, data_dir, char_to_idx, max_text_len=200, max_mel_len=1000):
#         self.data_dir = data_dir
#         self.char_to_idx = char_to_idx
#         self.max_text_len = max_text_len
#         self.max_mel_len = max_mel_len
#         self.sample_rate = 22050
#         self.n_mels = 80

#         # 存储所有样本的数据
#         self.data = []

#         stats = torch.load('mel_stats.pt')
#         self.mel_mean = stats['mean']
#         self.mel_std = stats['std']


#         metadata_path = os.path.join(data_dir, "metadata.csv")
#         print(f"Reading and preprocessing dataset...")

#         with open(metadata_path, encoding='utf-8') as f:
#             lines = f.readlines()

#         for line in lines:
#             parts = line.strip().split('|')
#             wav_id = parts[0]
#             wav_path = os.path.join(data_dir, "wavs", f"{wav_id}.wav")

#             if not os.path.exists(wav_path):
#                 continue
#             self.data.append(parts)

#         print(f"Loaded {len(self.data)} samples into memory.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         parts = self.data[idx]
#         wav_id = parts[0]
#         wav_path = os.path.join(self.data_dir, "wavs", f"{wav_id}.wav")
#         text = parts[-1].lower()  # 使用 normalized_text

#         # 文本编码
#         text_indices = [self.char_to_idx.get(c, self.char_to_idx["<PAD>"]) for c in text]
#         text_tensor = torch.tensor(text_indices, dtype=torch.long)

#         # 音频加载和 Mel 谱图计算
#         audio, sr = librosa.load(wav_path, sr=self.sample_rate)
#         mel_spec = librosa.feature.melspectrogram(
#             y=audio,
#             sr=sr,
#             n_mels=self.n_mels,
#             n_fft=1024,
#             hop_length=256,
#             win_length=1024,
#             fmin=0,
#             fmax=8000,
#             power=2.0
#         )
#         mel_spec = torch.tensor(mel_spec).T
#         mel_spec = torch.log(mel_spec + 1e-9)  # 对数压缩
#         mel_spec = (mel_spec - self.mel_mean) / self.mel_std

#         # 停止标志
#         stop_targets = torch.zeros(mel_spec.size(0))
#         stop_targets[-1] = 1  # 最后一帧为停止标记
        
#         return text_tensor, mel_spec, stop_targets

class LJSpeechDataset(Dataset):
    def __init__(self, data_dir, char_to_idx, max_text_len=200, max_mel_len=1000):
        self.data_dir = data_dir
        self.char_to_idx = char_to_idx
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
        self.sample_rate = 22050
        self.n_mels = 80

        # 存储所有样本的数据
        self.data = []

        metadata_path = os.path.join(data_dir, "metadata.csv")
        print(f"Reading and preprocessing dataset...")

        with open(metadata_path, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split('|')
            wav_id = parts[0]
            wav_path = os.path.join(data_dir, "wavs", f"{wav_id}.wav")

            if not os.path.exists(wav_path):
                continue
            self.data.append(parts)

        print(f"Loaded {len(self.data)} samples into memory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        parts = self.data[idx]
        wav_id = parts[0]
        wav_path = os.path.join(self.data_dir, "wavs", f"{wav_id}.wav")
        text = parts[-1].lower()  # 使用 normalized_text

        # 文本编码
        text_indices = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        text_tensor = torch.tensor(text_indices, dtype=torch.long)

        # 音频加载和 Mel 谱图计算
        # mel_spec = get_spectrograms(wav_path)
        mel_spec = np.load(wav_path[:-4] + '.pt.npy')
        mel_spec = torch.tensor(mel_spec)

        # 停止标志
        stop_targets = torch.zeros(mel_spec.size(0))
        stop_targets[-1] = 1  # 最后一帧为停止标记
        
        return text_tensor, mel_spec, stop_targets
        
# 3. 数据加载器
def get_dataloader(data_dir, batch_size=16):
    char_to_idx, _, vocab_size = create_vocab()
    dataset = LJSpeechDataset(data_dir, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda batch: collate_fn(batch))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
    #                         collate_fn=lambda batch: collate_fn(batch))
    return dataloader, vocab_size
    
from torch.nn.utils.rnn import pad_sequence
def generate_square_subsequent_mask(sz, device='cuda'):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def collate_fn(batch):
    texts, mels, stops = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=0)  # [B, T_text]
    mels = pad_sequence(mels, batch_first=True, padding_value=0.)   # [B, T_mel, n_mels]
    stops = pad_sequence(stops, batch_first=True, padding_value=1.) # [B, T_stop]

    src_key_padding_mask = (texts == 0)  # [B, T_text]
    tgt_key_padding_mask = (mels.sum(-1) == 0)  # [B, T_mel]

    # 构造 teacher forcing 输入：添加 zero frame 到开头，并右移
    B, T_mel, n_mels = mels.shape
    zero_frame = torch.zeros(B, 1, n_mels, device=mels.device)
    tgt_input = torch.cat([zero_frame, mels[:, :-1, :]], dim=1)  # [B, T_mel, n_mels]

    # 构造 stop token label（向量化）
    non_pad_lengths = (~tgt_key_padding_mask).sum(dim=1)    # [B]
    indices = non_pad_lengths - 1                           # 最后一帧
    batch_indices = torch.arange(B, device=mels.device)
    stop_token_labels = torch.zeros(B, T_mel, device=mels.device)
    stop_token_labels[batch_indices, indices] = 1


    return {
        'src': texts,
        'tgt_input': tgt_input,
        'tgt_label': mels,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'src_key_padding_mask': src_key_padding_mask,
        'stop_token_labels': stop_token_labels
    }

def audio_to_mel(wav_path, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000):
    audio, sr = librosa.load(wav_path, sr=sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    stats = torch.load('mel_stats.pt')
    mean = stats['mean']
    std = stats['std']

    mel_spectrogram = np.log(mel_spectrogram + 1e-9)  # 对数Mel频谱图
    mel_spectrogram = torch.tensor(mel_spectrogram.T)  # [n_frames, n_mels]
    mel_spectrogram = (mel_spectrogram - mean) / std

    return mel_spectrogram
def mel_to_audio(mel_outputs, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024):

    # mel_outputs = mel_outputs.cpu().numpy()
    # mel_outputs = np.exp(mel_outputs)
    # mel_outputs = np.maximum(mel_outputs, 1e-9)

    # 加载 mean & std
    stats = torch.load('mel_stats.pt')
    mean = stats['mean']
    std = stats['std']
    
    # 在函数中使用
    mel_outputs = mel_outputs.cpu().numpy()
    mel_outputs = mel_outputs * std + mean  # 去归一化
    mel_outputs = np.exp(mel_outputs)       # 指数还原
    
    # Reconstruct audio from Mel spectrogram using Griffin-Lim
    audio = librosa.feature.inverse.mel_to_audio(
        mel_outputs.T,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    return audio, sample_rate

if __name__ == "__main__":
    # data_dir = "/root/autodl-tmp/LJSpeech-1.1"
    wav_path = "/root/autodl-tmp/LJSpeech-1.1/wavs/LJ001-0001.wav"
    # # get_dataloader(data_dir)
    # char_to_idx, _, vocab_size = create_vocab()
    # dataset = LJSpeechDataset(data_dir, char_to_idx)
    # for i in range(len(dataset)):
    #     print(i)
    #     dataset[i]

    # exit(0)
    mel = audio_to_mel(wav_path)
    audio, sr = mel_to_audio(mel)
    torchaudio.save("output.wav", torch.tensor(audio).unsqueeze(0), sr)
    print("Audio saved as output.wav")

    # from tqdm import tqdm

    # def compute_mean_std(dataset):
    #     all_mels = []
    #     for i in tqdm(range(len(dataset))):
    #         _, mel, _ = dataset[i]  # 只取 mel spectrogram
    #         all_mels.append(mel.reshape(-1))  # 展平成一维
    #     all_mels = torch.cat(all_mels)
    #     mean = all_mels.mean().item()
    #     std = all_mels.std().item()
    #     return mean, std

    # # 使用示例
    # train_dataset = LJSpeechDataset(data_dir, char_to_idx)
    # _, mel, _ = train_dataset[0]
    # print("Sample min:", mel.min().item())
    # print("Sample max:", mel.max().item())
    # print("Sample mean:", mel.mean().item())
    # mean, std = compute_mean_std(train_dataset)
    # print(f"Mean: {mean}, Std: {std}")
    # torch.save({'mean': mean, 'std': std}, 'mel_stats.pt')