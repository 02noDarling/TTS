import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloader
from tacotron2 import Tacotron2

# 假设已定义Encoder, Decoder, Attention, Tacotron2类（见之前的代码）
# 这里仅贴出训练部分

def train(model, dataloader, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (text, mels, stop_targets, mel_lens) in enumerate(dataloader):
            text, mels, stop_targets = text.to(device), mels.to(device), stop_targets.to(device)
            
            optimizer.zero_grad()
            mel_outputs, stop_tokens = model(text, mels)

            # 裁剪mel_outputs到实际长度
            max_mel_len = torch.max(mel_lens).item()
            mel_outputs = mel_outputs[:, :max_mel_len]
            mels = mels[:, :max_mel_len]
            stop_tokens = stop_tokens[:, :max_mel_len]
            stop_targets = stop_targets[:, :max_mel_len]
            
            # 计算损失
            mel_loss = F.mse_loss(mel_outputs, mels)
            stop_loss = F.binary_cross_entropy(stop_tokens.squeeze(-1), stop_targets)
            loss = mel_loss + stop_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 1 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        # 保存模型
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/Users/hudaili/Desktop/VsCodeProjects/TTS/data/LJSpeech-1.1"  # 替换为你的LJSpeech数据集路径
    dataloader, vocab_size = get_dataloader(data_dir, batch_size=10)
    
    model = Tacotron2(vocab_size=vocab_size, hidden_size=512, n_mels=80).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train(model, dataloader, optimizer, device, epochs=1)