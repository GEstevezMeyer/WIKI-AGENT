import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, embed_dim: int, src_vocab_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(src_vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, dropout: float = 0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ScaledDotProduct(nn.Module):
    def __init__(self, dk: int):
        super().__init__()
        self.dk = dk

    def forward(self, V, K, Q, mask=None):
        x = torch.matmul(Q, K.transpose(-2, -1))
        x = x / math.sqrt(self.dk)

        if mask is not None:
            x = x.masked_fill(mask == 0, float('-inf'))

        x = nn.functional.softmax(x, dim=-1)
        return torch.matmul(x, V)


class Attention(nn.Module):
    def __init__(self, embed_dim: int, dk: int, dropout=0):
        super().__init__()
        self.V_linear = nn.Linear(embed_dim, dk)
        self.Q_linear = nn.Linear(embed_dim, dk)
        self.K_linear = nn.Linear(embed_dim, dk)
        self.ScaledDotProduct = ScaledDotProduct(dk)
        self.dropout = nn.Dropout(dropout)

    def forward(self, V, Q, K, mask=None):
        V = self.V_linear(V)
        Q = self.Q_linear(Q)
        K = self.K_linear(K)
        x = self.ScaledDotProduct(V, K, Q, mask=mask)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, number_heads: int, p_dropout: float = 0):
        super().__init__()
        assert embed_dim % number_heads == 0, 'embed_dim is not divisible by h'

        self.d_k = embed_dim // number_heads

        self.attentionHeads = nn.ModuleList([
            Attention(embed_dim, self.d_k, p_dropout)
            for _ in range(number_heads)
        ])
        self.O_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, V, Q, K, mask=None):
        x_array = [head(V, Q, K, mask=mask) for head in self.attentionHeads]
        x = torch.cat(x_array, dim=-1)
        return self.O_linear(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, d_ff: int = 0, p_dropout: float = 0):
        super().__init__()
        if d_ff == 0:
            d_ff = embed_dim * 4

        self.linear1 = nn.Linear(in_features=embed_dim, out_features=d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim: int, number_heads: int, p_dropout: float = 0, d_ff: int = 0):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, number_heads, p_dropout)
        self.ff = FeedForward(embed_dim, d_ff=d_ff, p_dropout=p_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None):
        x_res = x
        x = self.mha(x, x, x, mask=src_mask)
        x = x + x_res
        x = self.norm1(x)

        x_res = x
        x = self.ff(x)
        x = x + x_res
        x = self.norm2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, number_heads: int, p_dropout: float = 0, d_ff: int = 0):
        super().__init__()
        self.maskedmha = MultiHeadAttention(embed_dim, number_heads, p_dropout)
        self.mha = MultiHeadAttention(embed_dim, number_heads, p_dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim, d_ff, p_dropout)

    def forward(self, y, encoder_output, src_mask=None, tgt_mask=None):
        y_res = y
        y = self.maskedmha(y, y, y, mask=tgt_mask)
        y = y + y_res
        y = self.norm1(y)

        y_res = y
        y = self.mha(V=encoder_output, Q=y, K=encoder_output, mask=src_mask)
        y = y + y_res
        y = self.norm2(y)

        y_res = y
        y = self.ff(y)
        y = y + y_res
        y = self.norm3(y)

        return y


class Transformer(nn.Module):
    def __init__(self, number_layers: int, seq_len: int, src_vocab_size: int, tgt_vocab_size: int,
                 embed_dim: int, number_heads: int, p_dropout: float = 0, d_ff: int = 0):
        super().__init__()
        self.encoder_emb = InputEmbedding(embed_dim, src_vocab_size)
        self.decoder_emb = InputEmbedding(embed_dim, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_len, dropout=p_dropout)

        self.encoder_layers = nn.ModuleList([
            Encoder(embed_dim, number_heads, p_dropout=p_dropout, d_ff=d_ff)
            for _ in range(number_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_dim, number_heads, p_dropout, d_ff)
            for _ in range(number_layers)
        ])
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(p_dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()

        tgt_mask = tgt_padding_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_emb(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_emb(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output