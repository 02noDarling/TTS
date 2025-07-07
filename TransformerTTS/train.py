import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from TransformerTTS import *
import os
import hyperparams as hp

# 假设已定义Encoder, Decoder, Attention, Tacotron2类（见之前的代码）
# 这里仅贴出训练部分

def train(model, dataloader, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (texts, mels, stop_targets, mel_lengths) in enumerate(dataloader):
            texts, mels, stop_targets, mel_lengths = texts.to(device), mels.to(device), stop_targets.to(device), mel_lengths.to(device)
            
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
            
            # 停止标志损失
            stop_loss = F.binary_cross_entropy(stop_tokens, stop_targets.unsqueeze(-1), reduction='none')  # [batch, max_mel_len, 1]
            stop_loss = stop_loss * mask_stop  # 屏蔽填充部分
            stop_loss = stop_loss.sum() / mask_stop.sum()  # 平均有效帧

            # 计算损失
            loss = mel_loss + 10*stop_loss

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
    dataloader, vocab_size = get_dataloader(data_dir, batch_size=25)
    # model = Tacotron2(vocab_size=vocab_size, embed_size=512, hidden_size=512, n_mels=80).to(device)
    # if os.path.exists("/root/autodl-tmp/TTS/checkpoints.pt"):
    #     model.load_state_dict(torch.load("checkpoints.pt"))
    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # train(model, dataloader, optimizer, device, epochs=100)

    # 初始化模型
    model = TransformerTTS(vocab_size=vocab_size, n_mels=80).to(device)
    model_path = "/Users/hudaili/Desktop/VsCodeProjects/TTS/TransformerTTS/checkpoints.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("YYYYYYYYYYYYYY")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 损失函数（保持不变）
    def masked_mse_loss(pred, target, mask):
        non_pad_mask = ~mask
        pred = pred.masked_select(non_pad_mask.unsqueeze(-1))
        target = target.masked_select(non_pad_mask.unsqueeze(-1))
        return F.mse_loss(pred, target)
    
    def masked_l1_loss(pred, target, mask):
        non_pad_mask = ~mask
        pred = pred.masked_select(non_pad_mask.unsqueeze(-1))
        target = target.masked_select(non_pad_mask.unsqueeze(-1))
        return nn.L1Loss()(pred, target)

    def masked_bce_loss(pred, target, mask):
        non_pad_mask = ~mask  # 只计算非 padding 帧
        pred = pred.masked_select(non_pad_mask)
        target = target.masked_select(non_pad_mask)
        return F.binary_cross_entropy(pred, target)


    epochs = 10000
    global_step = 0

    def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
        lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            global_step += 1
            # if global_step < 400000:
            #     adjust_learning_rate(optimizer, global_step)
            
            src = data['src'].to(device)
            tgt_input = data['tgt_input'].to(device)
            tgt_label = data['tgt_label'].to(device)
            tgt_key_padding_mask = data['tgt_key_padding_mask'].to(device)
            src_key_padding_mask = data['src_key_padding_mask'].to(device)
            stop_token_labels = data['stop_token_labels'].to(device)  # 新增

            

            # 构造自注意力 mask（非必须，但可选）
            T_mel = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(T_mel, device=device)

            # 前向传播
            mel_output, mel_post, stop_token = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask
            )

            # Loss 分别计算
            loss_mel_pre = masked_l1_loss(mel_output, tgt_label, tgt_key_padding_mask)
            loss_mel_post = masked_l1_loss(mel_post, tgt_label, tgt_key_padding_mask)
            loss_stop = masked_bce_loss(stop_token.squeeze(-1), stop_token_labels, tgt_key_padding_mask)

            # first_loss = F.mse_loss(mel_output[:,0,:], tgt_label[:,0,:])

            # 合并 loss（可调权重）
            loss = loss_mel_pre + loss_mel_post + loss_stop 
            # loss = first_loss

            print(f"Mel loss Pre: {loss_mel_pre.item():.4f}, Mel loss Post: {loss_mel_post.item():.4f}, Stop loss: {loss_stop.item():.4f}, Total loss: {loss.item():.4f}")
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        # 保存模型
        torch.save(model.state_dict(), f"checkpoints.pt")