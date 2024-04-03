import torch as th

from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.positon_embedding = nn.Parameter(th.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.positon_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__() 

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # (batch_size, seq_len, dim)
        residual = x

        ## SELF ATTENTION

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residual

        residual = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * th.sigmoid(1.702 * x) # QuickGELU activation function

        x = self.linear_1(x)

        x += residual

        return x




class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList(
            [
                CLIPLayer(12, 768) for _ in range(12)
            ]
        )
        self.layernorm = nn.LayerNorm(768)


    def forward(self, token: th.LongTensor)-> th.FloatTensor:
        tokens = tokens.type(th.long)
        

        # ( batch_size, seq_len) -> ( batch_size, seq_len, dim) 
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (batch_size, seq_len, dim )
        output = self.layernorm(state)

        return output

