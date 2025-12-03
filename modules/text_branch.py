import torch
import torch.nn as nn
import safetensors
from modules.resampler import TextIDResampler

class MLPProjModel(torch.nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, inner_dim),
            torch.nn.GELU(),
            torch.nn.Linear(inner_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)

class TextBranch(nn.Module):
    def __init__(
        self,
        cross_attention_dim_1: int,
        cross_attention_dim_2: int,
        num_tokens: int = 2,
        face_recog_model_dim: int = 512,
        num_layers=4,
        dim_head=64,
    ) -> None:
        super().__init__()
        self.num_classes_1 = num_tokens * cross_attention_dim_1
        self.num_classes_2 = num_tokens * cross_attention_dim_2
        self.n_embeds = num_tokens
        self.num_layers = num_layers
        self.dim_head = dim_head

        self.cross_attention_dim_1 = cross_attention_dim_1
        self.cross_attention_dim_2 = cross_attention_dim_2

        self.proj_1 = MLPProjModel(face_recog_model_dim, face_recog_model_dim * 2, self.num_classes_1)
        self.proj_2 = MLPProjModel(face_recog_model_dim, face_recog_model_dim * 2, self.num_classes_2)
        self.norm_1 = nn.LayerNorm(self.cross_attention_dim_1)
        self.norm_2 = nn.LayerNorm(self.cross_attention_dim_2)

        self.text_resampler_1 = TextIDResampler(
            dim=self.cross_attention_dim_1,
            depth=num_layers,
            dim_head=dim_head,
            heads=self.cross_attention_dim_1 // dim_head,
            face_recog_model_dim=face_recog_model_dim,
            output_dim=self.cross_attention_dim_1,
            ff_mult=4,
        )
        self.text_resampler_2 = TextIDResampler(
            dim=self.cross_attention_dim_2,
            depth=num_layers,
            dim_head=dim_head,
            heads=self.cross_attention_dim_2 // dim_head,
            face_recog_model_dim=face_recog_model_dim,
            output_dim=self.cross_attention_dim_2,
            ff_mult=4,
        )

    def load_from_checkpoint(self, ckpt_path: str, device: str = "cpu"):
        if ckpt_path.split(".")[-1] == "safetensors":
            state_dict = {}
            with safetensors.safe_open(ckpt_path, framework="pt", device=device) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensors(k)
        else:
            state_dict = torch.load(ckpt_path, map_location=device)
        self.load_state_dict(state_dict)

    def forward(self, x, face_recog_model_last_hidden_state):
        x_1 = self.proj_1(x).reshape(-1, self.n_embeds, self.cross_attention_dim_1)
        x_2 = self.proj_2(x).reshape(-1, self.n_embeds, self.cross_attention_dim_2)
        x_1 = self.norm_1(x_1)
        x_2 = self.norm_2(x_2)

        last_hidden_state = face_recog_model_last_hidden_state.reshape(-1, 512, 7 * 7).transpose(
            1, 2
        )  # (batch, 512, 7, 7) to (batch, 49, 512)
        x_1 = self.text_resampler_1(x_1, last_hidden_state)
        x_2 = self.text_resampler_2(x_2, last_hidden_state)
        
        return x_1, x_2
