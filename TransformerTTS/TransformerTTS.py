import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from collections import OrderedDict
from text.symbols import symbols


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.alpha * self.pe[:, :x.size(1), :]

class EncoderPrenet(nn.Module):
    def __init__(self, embedding_size):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_size, embedding_size, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(embedding_size, embedding_size, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_size, embedding_size, kernel_size=5, padding=2)

        self.batch_norm1 = nn.BatchNorm1d(embedding_size)
        self.batch_norm2 = nn.BatchNorm1d(embedding_size)
        self.batch_norm3 = nn.BatchNorm1d(embedding_size)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, input_):
        input_ = self.embed(input_) 
        input_ = input_.transpose(1, 2) 
        input_ = self.dropout1(torch.relu(self.batch_norm1(self.conv1(input_)))) 
        input_ = self.dropout2(torch.relu(self.batch_norm2(self.conv2(input_)))) 
        input_ = self.dropout3(torch.relu(self.batch_norm3(self.conv3(input_)))) 
        input_ = input_.transpose(1, 2) 
        input_ = self.projection(input_) 

        return input_
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_prenet = EncoderPrenet(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_dropout = nn.Dropout(p=0.1)

    def forward(self, src, src_key_padding_mask=None):
        """
        src: [B, T]
        output: [B, T, D]
        """
        # x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.encoder_prenet(src)
        x = self.pos_encoder(x)
        x = self.pos_dropout(x)
        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

class DecoderPrenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(DecoderPrenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(p)),
             ('fc2', nn.Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out

class Decoder(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=6, dim_feedforward=2048, n_mels=80, dropout=0.1):
        super(Decoder, self).__init__()
        # self.mel_proj = nn.Linear(n_mels, embed_dim)
        self.decoder_prenet = DecoderPrenet(n_mels, embed_dim*2, embed_dim, p=0.2)
        self.norm = nn.Linear(embed_dim, embed_dim)
        self.pos_decoder = PositionalEncoding(embed_dim)
        self.pos_dropout = nn.Dropout(p=0.1)
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.mel_linear = nn.Linear(embed_dim, n_mels)
        self.stop_linear = nn.Linear(embed_dim, 1)

    def forward(self, memory, tgt, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        memory: [B, T_enc, D]
        tgt: [B, T_dec, n_mels]
        output: [B, T_dec, n_mels], [B, T_dec, 1]
        """
        # tgt_emb = self.mel_proj(tgt)
        tgt_emb = self.decoder_prenet(tgt)
        tgt_emb = self.norm(tgt_emb)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_emb = self.pos_dropout(tgt_emb)
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        mel_output = self.mel_linear(output)
        stop_token = torch.sigmoid(self.stop_linear(output))
        return mel_output, stop_token


class TransformerTTS(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, n_mels=80, dropout=0.1):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, num_encoder_layers, nhead, dim_feedforward, dropout)
        self.decoder = Decoder(embed_dim, nhead, num_decoder_layers, dim_feedforward, n_mels, dropout)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
        
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
        
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
        
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
        
            nn.Conv1d(embed_dim, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
        )

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        """
        src: [B, T_enc]
        tgt: [B, T_dec, n_mels]
        output: [B, T_dec, n_mels], [B, T_dec, n_mels], [B, T_dec, 1]
        """
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        B = src.size(0)
        mel_output, stop_token = self.decoder(
            memory,
            tgt,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # return mel_output, stop_token
        mel_post = mel_output + self.postnet(mel_output.transpose(1, 2)).transpose(1, 2)
        return mel_output, mel_post, stop_token

    