import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        
        pe = torch.zeros(seq_len, d_model)

       
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)



class ScaledDotProduct(nn.Module):
    def __init__(self, dk:int,mask:bool):
        super().__init__()
        self.dk = dk
        self.mask = mask
    def forward(self, V, K, Q):
        x = torch.matmul(Q, K.transpose(-2, -1))        
        x = x / math.sqrt(self.dk)

        if self.mask: 
            mask = torch.triu(torch.ones(x.shape[-2], x.shape[-1]), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            x+= mask.to(x.device)
        x = nn.functional.softmax(x, dim=-1)            
        return torch.matmul(x, V)


class Attention(nn.Module):
    def __init__(self, embed_dim:int, dk:int, dropout=0,mask = False):
        super().__init__()
        self.V_linear = nn.Linear(embed_dim, dk)
        self.Q_linear = nn.Linear(embed_dim, dk)
        self.K_linear = nn.Linear(embed_dim, dk)
        self.ScaledDotProduct = ScaledDotProduct(dk,mask)
        self.dropout = nn.Dropout(dropout)

    def forward(self,V,Q,K):
        V = self.V_linear(V)
        Q = self.Q_linear(Q)
        K = self.K_linear(K)
        x = self.ScaledDotProduct(V, K, Q)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, number_heads: int, p_dropout: float = 0,mask = False):
        super().__init__()
        assert embed_dim % number_heads == 0, 'embed_dim is not divisible by h'

        self.d_k = embed_dim // number_heads

        self.attentionHeads = nn.ModuleList([
            Attention(embed_dim, self.d_k, p_dropout,mask)
            for _ in range(number_heads)
        ])
        self.O_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self,V,Q,K):
        x_array = [head(V,Q,K) for head in self.attentionHeads]
        x = torch.cat(x_array, dim=-1)                
        return self.O_linear(x)


class FeedForward(nn.Module):
    def __init__(self,embed_dim:int,d_ff:int = 0,p_dropout:float = 0):
        super().__init__()
        if d_ff == 0:
            d_ff = embed_dim*4 

        self.linear1 = nn.Linear(in_features=embed_dim,out_features=d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(in_features=d_ff,out_features=embed_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self,embed_dim:int,number_heads:int,p_dropout:float = 0,d_ff:int = 0):
        super().__init__()

        self.mha = MultiHeadAttention(embed_dim,number_heads,p_dropout)
        self.ff = FeedForward(embed_dim,d_ff=d_ff,p_dropout=p_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        x_res = x
        x = self.mha(x,x,x)
        x = x + x_res
        x = self.norm1(x)
        x_res = x
        x = self.ff(x)
        x = x + x_res
        x = self.norm2(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self,embed_dim:int,number_heads:int,p_dropout:float = 0,d_ff:int = 0):
        super().__init__()

        self.maskedmha = MultiHeadAttention(embed_dim,number_heads,p_dropout,mask=True)
        self.mha = MultiHeadAttention(embed_dim,number_heads,p_dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim,d_ff,p_dropout)

    def forward(self,y,encoder_output):
        
            y_res = y
            y = self.maskedmha(y,y,y)
            y = y+y_res 
            y = self.norm1(y)
            y_res = y

            y = self.mha(V=encoder_output, Q=y, K=encoder_output)
            y = y + y_res
            y = self.norm2(y)
            y_res = y

            y = self.ff(y)
            y = y + y_res
            y = self.norm3(y)

            return y






if __name__ == "__main__":
    tensortest = torch.rand(1,30)
    encodertest = Encoder(30,3)
    decodertest = Decoder(30,3)
    encoder_output = encodertest(tensortest)
    decoder_output = decodertest(tensortest,encoder_output)

    print(decoder_output)


