# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)
    

class TextIDResampler(torch.nn.Module):
    def __init__(
        self,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        face_recog_model_dim=512,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(face_recog_model_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, id_embedding, face_embedding):
        face_embedding = self.proj_in(face_embedding)
        for attn, ff in self.layers:
            id_embedding = attn(face_embedding, id_embedding) + id_embedding
            id_embedding = ff(id_embedding) + id_embedding
        id_embedding = self.proj_out(id_embedding)
        return self.norm_out(id_embedding)
    
class ImageResampler(nn.Module):
    def __init__(
        self,
        dim=768,
        depth=6,
        dim_head=64,
        heads=16,
        face_recog_model_dims=[64,128,256,512],
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        self.proj_face_recog_model_layers = torch.nn.ModuleList([torch.nn.Linear(face_recog_model_dim, dim) for face_recog_model_dim in face_recog_model_dims])
        self.proj_out = torch.nn.Linear(dim, output_dim)
        torch.nn.init.zeros_(self.proj_out.weight)
        torch.nn.init.zeros_(self.proj_out.bias)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.face_recog_model_dims=face_recog_model_dims
        self.depth=depth
        self.dim=dim
        self.dim_head=dim_head
        self.heads=heads
        self.ff_mult=ff_mult
        
        self.init_layers()
    
    def init_layers(self):
        self.layers=self._init_layers(self.depth, len(self.face_recog_model_dims), self.dim, self.dim_head, self.heads, self.ff_mult)
        
    def _init_layers(self, depth, num_attn_layers, dim, dim_head, heads, ff_mult):
        layers=torch.nn.ModuleList([])
        for _ in range(depth):
            sub_layers=torch.nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads) for _ in range(num_attn_layers)])
            sub_layers.append(FeedForward(dim=dim, mult=ff_mult))
            layers.append(sub_layers)
        return layers
    
    def forward(self, id_embedding, face_recog_model_embedding):
        face_recog_model_embedding=[self.proj_face_recog_model_layers[i](face_recog_model_embedding[i]) for i in range(len(face_recog_model_embedding)-1,-1,-1)]
        for sub_layers in self.layers:
            for i,layer in enumerate(sub_layers):
                if i<len(sub_layers)-1:
                    id_embedding = layer(face_recog_model_embedding[i], id_embedding) + id_embedding
                else:
                    id_embedding = layer(id_embedding) + id_embedding
        id_embedding = self.proj_out(id_embedding)
        return self.norm_out(id_embedding)