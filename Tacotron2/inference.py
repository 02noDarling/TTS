import torch
import torchaudio
import librosa
import numpy as np
from tacotron2 import *
from dataset import *

# 推理函数
def infer(model, text, char_to_idx, device, max_len=1000):
    model.train()
    with torch.no_grad():
        # 文本预处理
        text = text.lower()
        text_indices = [char_to_idx.get(c, char_to_idx["<PAD>"]) for c in text]
        # text_indices = text_indices[:200] + [char_to_idx["<PAD>"]] * (200 - len(text_indices))
        text_tensor = torch.tensor([text_indices], dtype=torch.long).to(device)  # [1, text_len]
        
        # 推理生成 Mel
        _, mel_outputs_post, stop_tokens = model(text_tensor, max_len=max_len)  # [1, n_frames, n_mels], [1, n_frames, 1]
        print("Generated Mel shape:", mel_outputs_post.shape)
        
        # # 裁剪到停止点
        # stop_tokens = stop_tokens.squeeze(-1)  # [1, n_frames]
        # stop_idx = torch.where(stop_tokens > 0.5)[1]
        # if len(stop_idx) > 0:
        #     mel_outputs = mel_outputs[:, :stop_idx[0] + 1, :]  # 裁剪到第一个停止点
        # print(mel_outputs.shape)
        return mel_outputs_post.squeeze(0)  # [n_frames, n_mels]

# 主推理流程
if __name__ == "__main__":
    # 参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = 22050
    n_mels = 80
    n_fft = 2048
    hop_length = 256
    win_length = 2048
    max_len = 1000
    
    # 加载词汇表
    char_to_idx, idx_to_char, vocab_size = create_vocab()
    
    # 加载模型（需替换为你的模型路径）
    model = Tacotron2(vocab_size=vocab_size, n_mels=n_mels).to(device)
    model.load_state_dict(torch.load("checkpoints.pt", map_location=device))
    
    # 输入文本
    text = "in being comparatively modern."
    
    # 推理
    mel = infer(model, text, char_to_idx, device, max_len)
    
    # 转换为音频
    audio, sr = mel_to_audio(mel, sample_rate, n_mels, n_fft, hop_length, win_length)
    
    # 保存音频
    torchaudio.save("output_infer.wav", torch.tensor(audio).unsqueeze(0), sr)
    print("推理音频已保存为 output_infer.wav")