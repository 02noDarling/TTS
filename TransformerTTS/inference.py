import torch
import torchaudio
import librosa
import numpy as np
from TransformerTTS import *
from dataset import *
from text import *
from utils import mel_to_audio

# 推理函数
# def generate(model, text, device, max_len=1000):
#     model.eval()
#     with torch.no_grad():
#         text = text.lower()
#         text_indices = [char_to_idx.get(c, char_to_idx["<PAD>"]) for c in text]
#         # text_indices = text_indices[:200] + [char_to_idx["<PAD>"]] * (200 - len(text_indices))
#         src = torch.tensor([text_indices], dtype=torch.long).to(device)  # [1, text_len]
#         # src = text.unsqueeze(0).to(device)  # [1, T_text]
#         src_key_padding_mask = (src == 0)

#         memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)

#         # 初始化输入为 zero frame
#         B = 1
#         n_mels = model.decoder.mel_linear.out_features
#         tgt_input = torch.zeros(B, 1, n_mels, device=device)

#         generated = []

#         for t in range(max_len):
#             tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)
#             mel_output, stop_token = model.decoder(
#                 memory,
#                 tgt_input,
#                 tgt_mask=tgt_mask,
#                 memory_key_padding_mask=src_key_padding_mask
#             )
#             mel_output = mel_output + model.postnet(mel_output.transpose(1, 2)).transpose(1, 2)
#             last_mel = mel_output[:, -1:, :]  # [1, 1, n_mels]
#             stop_flag = (stop_token[:, -1, 0] > 0.5)

#             generated.append(last_mel)
#             tgt_input = torch.cat([tgt_input, last_mel], dim=1)

#             if stop_flag:
#                 break

#         generated = torch.cat(generated, dim=1)
#         print(generated.shape)
#         return generated.squeeze(0)

# def generate(model, text, device, max_len=1000, stop_threshold=0.5):
#     model.eval()
#     with torch.no_grad():
#         # 文本编码
#         text = text.lower()
#         text_indices = [char_to_idx.get(c, char_to_idx["<PAD>"]) for c in text]
#         src = torch.tensor([text_indices], dtype=torch.long).to(device)  # [1, T_text]
#         src_key_padding_mask = (src == 0)

#         # 编码器只运行一次
#         memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)

#         # 初始化 decoder 输入为 zero frame
#         B = 1
#         n_mels = model.decoder.mel_linear.out_features
#         tgt_input = torch.zeros(B, 1, n_mels, device=device)  # [1, 1, n_mels]

#         generated = []
#         stop_flag = False

#         for t in range(max_len):
#             # 构造自注意力 mask
#             tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)

#             # 解码器前向传播
#             mel_output, stop_token = model.decoder(
#                 memory,
#                 tgt_input,
#                 tgt_mask=tgt_mask,
#                 memory_key_padding_mask=src_key_padding_mask
#             )

#             # PostNet refine
#             mel_post = mel_output + model.postnet(mel_output.transpose(1, 2)).transpose(1, 2)

#             # 获取最后一帧
#             last_mel = mel_post[:, -1:, :]  # [1, 1, n_mels]
#             generated.append(last_mel)

#             # 更新 decoder 输入
#             tgt_input = torch.cat([tgt_input, last_mel], dim=1)

#             # Stop token 判断（连续两次超过阈值才停止）
#             if stop_token[:,-1:,:].squeeze() > stop_threshold:
#                 print(f"Stop token triggered at step {t}")
#                 break

#         # 拼接所有生成的帧
#         generated = torch.cat(generated, dim=1)
#         print("Generated mel shape:", generated.shape)
#         generated = generated + model.postnet(generated.transpose(1, 2)).transpose(1, 2)
#         return generated.squeeze(0)  # [T_mel, n_mels]
        
# def generate(model, text, device, max_len=1000, stop_threshold=0.5):
#     model.eval()
#     with torch.no_grad():
#         # 文本编码
#         text = text.lower()
#         text_indices = [char_to_idx.get(c, char_to_idx["<PAD>"]) for c in text]
#         src = torch.tensor([text_indices], dtype=torch.long).to(device)  # [1, T_text]
#         src_key_padding_mask = (src == 0)

#         # 编码器只运行一次
#         memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)

#         # 初始化 decoder 输入为 zero frame
#         B = src.size(0)
#         n_mels = model.decoder.mel_linear.out_features
#         tgt_input = torch.zeros(B, 1, n_mels, device=device)  # [1, 1, n_mels]

#         generated_outputs = []  # 存储原始 decoder 输出

#         for t in range(max_len):
#             # 构造自注意力 mask
#             tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)

#             # 解码器前向传播
#             mel_output, stop_token = model.decoder(
#                 memory,
#                 tgt_input,
#                 tgt_mask=tgt_mask,
#                 memory_key_padding_mask=src_key_padding_mask
#             )

#             # 获取最后一帧作为下一轮输入 ❌ 不经过 PostNet
#             last_mel = mel_output[:, -1:, :]  # [1, 1, n_mels]
#             generated_outputs.append(last_mel)

#             # 更新 decoder 输入
#             tgt_input = torch.cat([tgt_input, last_mel], dim=1)

#             # Stop token 判断
#             if stop_token[0, -1] > stop_threshold:
#                 print(f"Stop token triggered at step {t}")
#                 break

#         # 拼接所有生成的帧
#         all_mel_outputs = torch.cat(generated_outputs, dim=1)  # [1, T_dec, n_mels]

#         # 最后统一通过 PostNet 进行 refine ✅
#         mel_post = all_mel_outputs + model.postnet(all_mel_outputs.transpose(1, 2)).transpose(1, 2)

#         return mel_post.squeeze(0)  # [T_mel, n_mels]

def generate(model, text, device, max_len=1000, stop_threshold=0.5):
    model.eval()
    with torch.no_grad():
        # 文本编码
        text = text.lower()
        # text_indices = [char_to_idx.get(c, char_to_idx["<PAD>"]) for c in text]
        # src = torch.tensor([text_indices], dtype=torch.long).to(device)  # [1, T_text]
        text_indices = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        src = torch.tensor([text_indices], dtype=torch.long)
        print(src)
        src_key_padding_mask = (src == 0)

        # 初始化 decoder 输入为 zero frame
        B = src.size(0)
        n_mels = 80  # 替换为你自己的 n_mels
        tgt_input = torch.zeros(B, 1, n_mels, device=device)  # [1, 1, n_mels]

        generated_outputs = []

        for step in range(max_len):
            # 直接调用 forward 函数（训练和推理共享）
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)
            mel_output, mel_post, stop_token = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask
            )

            last_frame = mel_output[:, -1:, :]  # [B, 1, n_mels]
            # last_frame = label[step, :].unsqueeze(0).unsqueeze(0)  # [B, 1, n_mels]
            # print(last_frame.shape)
            generated_outputs.append(last_frame)

            # 更新 decoder 输入
            tgt_input = torch.cat([tgt_input, last_frame], dim=1)

            # Stop Token 判断
            # print(stop_token.shape)
            # if step < label.shape[0]:
            #     print(nn.L1Loss()(last_frame.squeeze(0), label[step,:]))
            # print(F.mse_loss(last_frame.squeeze(0), label[step,:]))
            if stop_token[0, -1, 0] > stop_threshold:
                print(f"Stop token triggered at step {step}")
                break

        # 拼接所有生成的帧
        # all_mel_outputs = torch.cat(generated_outputs, dim=1)  # [1, T_dec, n_mels]

        # # 最后再统一加 PostNet
        # mel_post = all_mel_outputs + model.postnet(all_mel_outputs.transpose(1, 2)).transpose(1, 2)

        return mel_post.squeeze(0)  # [T_mel, n_mels]
        


# 主推理流程
if __name__ == "__main__":
    # 参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 1000
    
    # 加载词汇表
    char_to_idx, idx_to_char, vocab_size = create_vocab()
    
    # 加载模型（需替换为你的模型路径）
    model_path = r"D:\VsCodeProjects\TTS\TransformerTTS\checkpoints.pt"
    model = TransformerTTS(vocab_size=vocab_size, n_mels=80).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 输入文本
    text = "Transformer is a great deep learning model"
    text = text.lower()
    # wav_path = "/root/autodl-tmp/LJSpeech-1.1/wavs/LJ001-0008.wav"
    # mel_label = audio_to_mel(wav_path)
    # mel_label = get_spectrograms(wav_path)
    
    # 推理
    # mel_label = torch.tensor(mel_label)
    mel = generate(model, text, device, max_len=max_len).to(device)
    # mel = mel[:max_len,:].to(device)
    # mel_label = mel_label[:max_len,:].to(device)
    # print(mel.shape)
    # print(mel_label.shape)

    # max_len = min(mel_label.shape[0], mel.shape[0])
    # print(mel.shape, mel_label.shape)
    # mel_label = torch.tensor(mel_label)
    # print(nn.L1Loss()(mel[:max_len,:], mel_label[:max_len,:]))
    # print(F.mse_loss(mel[:max_len,:], mel_label[:max_len,:]))
    
    # 转换为音频
    wav = mel_to_audio(mel.squeeze(0).cpu().numpy(), output_wav_path="output_infer.wav")
    
    # audio, sr = mel_to_audio(mel)
    
    # # 保存音频
    # torchaudio.save("output_infer.wav", torch.tensor(audio).unsqueeze(0), sr)
    print("推理音频已保存为 output_infer.wav")