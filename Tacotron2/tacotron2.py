import torch
import torch.nn as nn
import torch.nn.functional as F

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv1d(embed_size, hidden_size, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_size]
        x = x.transpose(1, 2)  # [batch, embed_size, seq_len]
        x = F.relu(self.conv(x))  # [batch, hidden_size, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_size]
        outputs, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        return outputs

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]
        query = self.query_layer(decoder_hidden).unsqueeze(1)  # [batch, 1, hidden_size]
        keys = self.key_layer(encoder_outputs)  # [batch, seq_len, hidden_size]
        energy = self.energy_layer(torch.tanh(query + keys))  # [batch, seq_len, 1]
        attention_weights = F.softmax(energy, dim=1)  # [batch, seq_len, 1]
        context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)  # [batch, 1, hidden_size]
        return context.squeeze(1), attention_weights

# 解码器
class Decoder(nn.Module):
    def __init__(self, hidden_size, n_mels):
        super(Decoder, self).__init__()
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.mel_linear = nn.Linear(hidden_size * 2, n_mels)
        self.stop_linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, memory, prev_mel, hidden):
        # prev_mel: [batch, n_mels]
        prenet_out = self.prenet(prev_mel)  # [batch, hidden_size]
        context, attn_weights = self.attention(hidden[0][-1], memory)  # [batch, hidden_size]
        lstm_input = torch.cat([prenet_out, context], dim=-1).unsqueeze(1)  # [batch, 1, hidden_size*2]
        lstm_out, hidden = self.lstm(lstm_input, hidden)  # [batch, 1, hidden_size]
        lstm_out = lstm_out.squeeze(1)  # [batch, hidden_size]
        concat_out = torch.cat([lstm_out, context], dim=-1)  # [batch, hidden_size*2]
        mel_out = self.mel_linear(concat_out)  # [batch, n_mels]
        stop_token = torch.sigmoid(self.stop_linear(concat_out))  # [batch, 1]
        return mel_out, stop_token, hidden, attn_weights

# Tacotron 2 模型
class Tacotron2(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512, n_mels=80):
        super(Tacotron2, self).__init__()
        self.n_mels = n_mels
        self.encoder = Encoder(vocab_size, embed_size, hidden_size)
        self.decoder = Decoder(hidden_size, n_mels)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_size, n_mels, kernel_size=5, padding=2)
        )
        

    def forward(self, text, mels=None, max_len=1000):
        memory = self.encoder(text)  # [batch, seq_len, hidden_size]
        batch_size = text.size(0)
        mel_outputs = []
        stop_tokens = []
        hidden = None
        prev_mel = torch.zeros(batch_size, self.n_mels).to(text.device)
        use_teacher_forcing = mels is not None

        # 初始化hidden
        if hidden is None:
            h = torch.zeros(1, batch_size, self.decoder.lstm.hidden_size).to(text.device)
            c = torch.zeros(1, batch_size, self.decoder.lstm.hidden_size).to(text.device)
            hidden = (h, c)

        for t in range(max_len):
            mel_out, stop_token, hidden, attn_weights = self.decoder(memory, prev_mel, hidden)
            mel_outputs.append(mel_out)
            stop_tokens.append(stop_token)
            if not use_teacher_forcing:
                prev_mel = mel_out
                if torch.all(stop_token > 0.5):  # 停止条件
                    break
            elif t < mels.shape[1]:
                    prev_mel = mels[:, t, :]
        
        mel_outputs = torch.stack(mel_outputs, dim=1)  # [batch, max_len, n_mels]
        stop_tokens = torch.stack(stop_tokens, dim=1)  # [batch, max_len, 1]
        mel_outputs_post = mel_outputs + self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        return mel_outputs, mel_outputs_post, stop_tokens

# 示例使用
if __name__ == "__main__":
    # 假设数据集已准备好
    vocab_size = 50  # 音素或字符的词汇表大小
    model = Tacotron2(vocab_size=vocab_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 假设 dataloader 提供 (text, mels, stop_targets)
    # dataloader = ...
    for epoch in range(100):
        avg_loss = train(model, dataloader, optimizer, model.device)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")