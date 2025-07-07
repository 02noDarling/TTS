import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloader
from tacotron2 import Tacotron2
import os
from dataset import *

# 假设已定义Encoder, Decoder, Attention, Tacotron2类（见之前的代码）
# 这里仅贴出训练部分

def train(model, dataloader, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (texts, mels, stop_targets, text_lengths, mel_lengths) in enumerate(dataloader):
            texts, mels, stop_targets, text_lengths, mel_lengths = texts.to(device), mels.to(device), stop_targets.to(device), text_lengths.to(device), mel_lengths.to(device)
            
            optimizer.zero_grad()
            mel_outputs, mel_outputs_post, stop_tokens  = model(texts, mels)

            # 创建掩码
            batch_size, max_mel_len, n_mels = mels.shape

            mel_outputs = mel_outputs[:, :max_mel_len]
            mel_outputs_post = mel_outputs_post[:, :max_mel_len]
            stop_tokens = stop_tokens[:, :max_mel_len]
            
            mask = torch.arange(max_mel_len, device=device)[None, :] < mel_lengths[:, None]  # [batch, max_mel_len
            mask_mel = mask.unsqueeze(-1).expand(-1, -1, n_mels)  # [batch, max_mel_len, n_mels]
            mask_stop = mask.unsqueeze(-1)  # [batch, max_mel_len, 1]
            
            # Mel 损失
            mel_loss_pre = F.mse_loss(mel_outputs, mels, reduction='none')  # [batch, max_mel_len, n_mels]
            mel_loss_pre = mel_loss_pre * mask_mel  # 屏蔽填充部分
            mel_loss_pre = mel_loss_pre.sum() / mask_mel.sum()  # 平均有效帧

            mel_loss_post = F.mse_loss(mel_outputs_post, mels, reduction='none')  # [batch, max_mel_len, n_mels]
            mel_loss_post = mel_loss_post * mask_mel  # 屏蔽填充部分
            mel_loss_post = mel_loss_post.sum() / mask_mel.sum()  # 平均有效帧

            mel_loss = mel_loss_pre + mel_loss_post
            

            print(stop_tokens)
            print(stop_targets)
            # 停止标志损失
            stop_loss = F.binary_cross_entropy(stop_tokens, stop_targets.unsqueeze(-1), reduction='none')  # [batch, max_mel_len, 1]
            stop_loss = stop_loss * mask_stop  # 屏蔽填充部分
            stop_loss = stop_loss.sum() / mask_stop.sum()  # 平均有效帧

            # 计算损失
            loss = mel_loss + stop_loss

            print(f"Mel loss pre: {mel_loss_pre.item():.4f}, Mel loss post: {mel_loss_post.item():.4f}, Stop loss: {stop_loss.item():.4f}, Total loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # if batch_idx % 1 == 0:
            #     print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        # 保存模型
        torch.save(model.state_dict(), f"checkpoints.pt")

# 主程序
if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/Users/hudaili/Desktop/VsCodeProjects/TTS/data/LJSpeech-1.1"  # 替换为你的LJSpeech数据集路径
    dataloader, vocab_size = get_dataloader(data_dir, batch_size=1)
    
    model = Tacotron2(vocab_size=vocab_size, embed_size=512, hidden_size=512, n_mels=80).to(device)
    if os.path.exists("checkpoints.pt"):
        model.load_state_dict(torch.load("checkpoints.pt", map_location=device))
        print("YYYYYYYYYYYYYYY")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    train(model, dataloader, optimizer, device, epochs=100)