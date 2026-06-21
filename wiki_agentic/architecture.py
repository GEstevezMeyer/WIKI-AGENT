import torch
import torch.nn as nn
import math

class ScaledDotProduct(nn.Module):
    def __init__(self, dk):
        super().__init__()
        self.dk = dk

    def forward(self, V, K, Q):
        x = torch.matmul(Q, K.transpose(-2, -1))        
        x = x / math.sqrt(self.dk)                      
        x = nn.functional.softmax(x, dim=-1)            
        return torch.matmul(x, V)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dk, dropout=0):
        super().__init__()
        self.V_linear = nn.Linear(embed_dim, embed_dim)
        self.Q_linear = nn.Linear(embed_dim, embed_dim)
        self.K_linear = nn.Linear(embed_dim, embed_dim)
        self.ScaledDotProduct = ScaledDotProduct(dk)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        V = self.V_linear(x)
        Q = self.Q_linear(x)
        K = self.K_linear(x)
        x = self.ScaledDotProduct(V, K, Q)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, number_heads: int, p_dropout: float = 0):
        super().__init__()
        assert embed_dim % number_heads == 0, 'd_model is not divisible by h'

        self.d_k = embed_dim // number_heads

        self.attentionHeads = nn.ModuleList([
            SelfAttention(embed_dim, self.d_k, p_dropout)
            for _ in range(number_heads)
        ])
        self.O_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x_array = [head(x) for head in self.attentionHeads]
        x = torch.cat(x_array, dim=-1)                  
        return self.O_linear(x)


