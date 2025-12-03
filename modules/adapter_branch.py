import torch
import torch.nn as nn
from modules.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor

from .resampler import ImageResampler

class AdapterBranch(nn.Module):
    def __init__(
        self, 
        unet, 
        hidden_dim = 1024, 
        dim_head = 64, 
        num_tokens = 16, 
        num_layers = 6, 
        face_recog_model_dims = [64+128+256,512], 
        normalize_adapter_output = False,
        ff_mult:int=4,
        scale_factor:float=1.0
    ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_tokens=num_tokens
        self.num_layers=num_layers
        self.dim_head=dim_head
        self.face_recog_model_dims=face_recog_model_dims
        self.normalize_adapter_output=normalize_adapter_output
        self.ff_mult=ff_mult
        self.unet_cross_attention_dim=unet.config.cross_attention_dim
        
        self.latents=nn.Parameter(torch.randn(1, num_tokens, hidden_dim) / hidden_dim**0.5)
        
        self.init_resampler()
        self.init_adapter_modules(unet, normalize_adapter_output, scale_factor=scale_factor)
    
    def init_resampler(self):
        self.adapter_resampler=ImageResampler(
            dim=self.hidden_dim,
            depth=self.num_layers,
            dim_head=self.dim_head,
            heads=self.hidden_dim//self.dim_head,
            face_recog_model_dims=self.face_recog_model_dims,
            output_dim=self.unet_cross_attention_dim,
            ff_mult=self.ff_mult,
        )
    
    def init_adapter_modules(self, unet, normalize_adapter_output=False, scale_factor:float=1.0):
        # init adapter modules
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=self.num_tokens, normalize_adapter_output=normalize_adapter_output, scale=scale_factor)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        
    def proj_face_recog_model_hidden_states(self, face_recog_model_hidden_states):
        face_recog_model_embeds=[]
        for embed in face_recog_model_hidden_states[:-1]:
            embed=nn.functional.interpolate(embed,size=(16,16),mode='bilinear',align_corners=False) # (barch, channel, 16, 16)
            embed=embed.permute(0,2,3,1) # (barch, 16, 16, channel)
            face_recog_model_embeds.append(embed.reshape(embed.shape[0],16*16,embed.shape[-1])) # (barch, 256, channel)
            
        face_recog_model_last_hidden_state=face_recog_model_hidden_states[-1]
        bs,channels,height,weight=face_recog_model_last_hidden_state.shape
        face_recog_model_last_hidden_state=face_recog_model_last_hidden_state.reshape(bs,channels,height*weight).transpose(1,2)
        
        return [torch.cat(face_recog_model_embeds,dim=-1),face_recog_model_last_hidden_state]
    
    def forward(self, face_recog_model_hidden_states):
        bs = face_recog_model_hidden_states[0].shape[0]
        latents=self.latents.expand(bs,-1,-1)
        
        face_recog_model_hidden_states_reshaped=self.proj_face_recog_model_hidden_states(face_recog_model_hidden_states)
        
        image_embeds=self.adapter_resampler(latents, face_recog_model_hidden_states_reshaped)
        return image_embeds