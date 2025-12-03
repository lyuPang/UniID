import torch.nn as nn
import torch

from .adapter_branch import AdapterBranch
from .text_branch import TextBranch
from .insightface_iresnet import iresnet100

def apply_mask(tensor:torch.tensor, mask:torch.tensor):
    mask_expanded = mask.view(*mask.shape, *([1] * (tensor.dim() - 1)))
    tensor=torch.masked_fill(tensor, mask_expanded, 0)
    return tensor

def load_face_recogniztion_model(
    pretrained_face_recog_model_path: str=None,
    freeze_model: bool = True,
) -> nn.Module:
    model = iresnet100()
    
    if pretrained_face_recog_model_path is not None:
        model.load_state_dict(torch.load(pretrained_face_recog_model_path,weights_only=True))
    if freeze_model:
        model.requires_grad_(False)
        model.eval()
    
    return model


class UniID(nn.Module):
    def __init__(
        self,
        unet,
        text_encoder_hidden_size=768,
        text_encoder_2_hidden_size=1280,
        num_text_branch_tokens:int=4,
        num_adapter_tokens:int=16,
        load_text_branch_from_checkpoint:str=None,
        adapter_hidden_dim:int=1024,
        normalize_adapter_output:bool=False,
        load_adapter_branch_from_checkpoint:str=None,
        drop_text_branch:bool=False,
        drop_adapter:bool=False,
    ):
        super().__init__()
        self.drop_text_branch=drop_text_branch
        self.drop_adapter=drop_adapter
        
        self.text_branch=TextBranch(
            cross_attention_dim_1=text_encoder_hidden_size,
            cross_attention_dim_2=text_encoder_2_hidden_size,
            num_tokens=num_text_branch_tokens,
            face_recog_model_dim=512,
            num_layers=4,
            dim_head=64,
        ) if not drop_text_branch else None
        if self.text_branch is not None and load_text_branch_from_checkpoint is not None:
            state_dict=torch.load(load_text_branch_from_checkpoint, weights_only=True)
            self.text_branch.load_state_dict(state_dict)
        self.adapter_branch=AdapterBranch(
            unet=unet,
            hidden_dim=adapter_hidden_dim,
            num_tokens=num_adapter_tokens,
            normalize_adapter_output=normalize_adapter_output,
            dim_head=64,
            num_layers=6,
            face_recog_model_dims=[64+128+256,512],
        ) if not drop_adapter else None
        if self.adapter_branch is not None and load_adapter_branch_from_checkpoint is not None:
            state_dict=torch.load(load_adapter_branch_from_checkpoint, weights_only=True)
            self.adapter_branch.load_state_dict(state_dict)
    
    def freeze_text_branch(self):
        if self.text_branch is not None:
            self.text_branch.requires_grad_(False)
            self.text_branch.eval()
    
    def freeze_adapter(self):
        if self.adapter_branch is not None:
            self.adapter_branch.requires_grad_(False)
            self.adapter_branch.eval()
        
    def forward(self, face_recog_model_outputs, face_recog_model_hidden_states,text_branch_masks=None, adapter_branch_masks=None):
        if not self.drop_text_branch:
            # apply mask
            if text_branch_masks is not None:
                with torch.no_grad():
                    face_recog_model_outputs_for_text_branch=apply_mask(face_recog_model_outputs, text_branch_masks)
                    face_recog_model_hidden_states_for_text_branch=[apply_mask(hidden_state, text_branch_masks) for hidden_state in face_recog_model_hidden_states]
            else:
                face_recog_model_outputs_for_text_branch=face_recog_model_outputs
                face_recog_model_hidden_states_for_text_branch=face_recog_model_hidden_states
            text_embeds=self.text_branch(face_recog_model_outputs_for_text_branch, face_recog_model_hidden_states_for_text_branch[-1])
        else:
            text_embeds=None

        if not self.drop_adapter:
            # apply mask
            if adapter_branch_masks is not None:
                with torch.no_grad():
                    face_recog_model_outputs=apply_mask(face_recog_model_outputs, adapter_branch_masks)
                    face_recog_model_hidden_states=[apply_mask(hidden_state, adapter_branch_masks) for hidden_state in face_recog_model_hidden_states]
            image_embeds=self.adapter_branch(face_recog_model_hidden_states)
        else:
            image_embeds=None
        
        return text_embeds,image_embeds