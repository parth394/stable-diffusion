import torch as th
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x: th.Tensor, causal_mask=True):
        # x is coming from VAE_Attention Block
        # x: (Batch_Size, height * width, feature) 
        # x: (Batch_Size, seq, dim) 
        # This new names are for common naming conventions of attention blocks

        input_shape = x.shape
        batch_size, seq_length, d_embed = input_shape

        # Basically d_embed = n_heads * d_head
        interim_shape = ( batch_size, seq_length, self.n_heads, self.d_head)

        # (batch_size, seq, dim) -> (batch_size, seq, 3 * dim) -> 3 * (batch_size, seq, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq, dim) -> (batch_size, seq, h, dim/ h ) -> (batch_size, h, seq, dim/h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # q * K_T
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = th.ones_like(weight, dtype=th.bool).triu(1)
            weight.masked_fill(mask, -th.inf)

        # (q * K_T)/sqrt(d_head)
        weight /= math.sqrt(self.d_heads)

        # applying softmax
        weight = F.softmax(weight, dim=-1)

        # (batch_size, h, seq, seq) @ ( batch_size, h, seq, dim/h ) ->  ( batch_size, h, seq, dim/h ) 
        output = weight @ v

        # ( batch_size, h, seq, dim/h )  -> ( batch_size, seq, h, dim/h ) 
        output = output.transpose(1, 2)

        # ( batch_size, seq, h, dim/h ) -> (batch_size, seq, dim) 
        output = output.reshape(input_shape)

        # (batch_size, seq, dim) -> (batch_size, seq, dim)
        output = self.out_proj(output)

        # ( batch_size, seq, dim )
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_prj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_head = n_heads
        self.d_head = d_embed//n_heads

    
    def forward(self, x, y):
        # x (latent): (batch_size, seq_len_q, dim_q )
        # y (context): (batch_size, seq_len_kv, dim_kv)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_head, self.d_head)
        
        # (batch_size, seq_len_q, dim_q ) -> (batch_size, seq_len_q, dim_q )
        q = self.q_proj(x)
        # (batch_size, seq_len_kv, dim_q ) -> (batch_size, seq_len_kv, dim_q )
        k = self.k_proj(y)
        # (batch_size, seq_len_kv, dim_q ) -> (batch_size, seq_len_kv, dim_q )
        v = self.v_proj(y)

        # (batch_size, seq_len_q, dim_q ) -> batch_size, seq_len_q, n_head, dim_q/n_head) -> (batch_size, n_head, seq_len_q, dim_q/n_head)
        q = q.view(interim_shape).transpose(1, 2)
        # (batch_size, seq_len_kv, dim_q ) -> batch_size, seq_len_kv, n_head, dim_q/n_head) -> (batch_size, n_head, seq_len_kv, dim_q/n_head)
        k = k.view(interim_shape).transpose(1, 2)
        # (batch_size, seq_len_kv, dim_q ) -> batch_size, seq_len_kv, n_head, dim_q/n_head) -> (batch_size, n_head, seq_len_kv, dim_q/n_head)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, n_head, seq_len_q, dim_q/n_head) @ (batch_size, n_head, dim_q/n_head, seq_len_kv) -> (batch_size, n_head, seq_len_q, seq_len_kv)
        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, n_head, seq_len_q, seq_len_kv) @ (batch_size, n_head, seq_len_kv, dim_q/n_head) -> (batch_size, n_head, seq_len_q, dim_q/n_head)
        output = weight @ v

        # (batch_size, n_head, seq_len_q, dim_q/n_head) -> (batch_size, seq_len_q, n_head, dim_q/n_head)
        output = output.transpose(1, 2).contigous()

        # (batch_size, seq_len_q, n_head, dim_q/n_head) -> (batch_size, seq_len_q, dim_q)
        output = output.view(input_shape)

        # (batch_size, seq_len_q, dim_q) -> (batch_size, seq_len_q, dim_q)
        output = self.out_proj(output)

        return output
