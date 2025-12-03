import subprocess
import os
import random

import torch
from tqdm import tqdm
from transformers.models.clip import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from PIL import Image
import numpy as np

@torch.no_grad()
def get_celeb_col_mean(
    celeb_path : str,
    tokenizer_1 : CLIPTokenizer,
    tokenizer_2 : CLIPTokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    n_column: int=3,
):
    """
    Compute the mean embeddings of the columns in the celeb dataset.

    Args:
    - celeb_path (str): The path to the celeb dataset.
    - tokenizer_1 (CLIPTokenizer): The tokenizer for the first text encoder.
    - tokenizer_2 (CLIPTokenizer): The tokenizer for the second text encoder.
    - text_encoder_1 (CLIPTextModel): The first text encoder.
    - text_encoder_2 (CLIPTextModelWithProjection): The second text encoder.
    - n_column (int, optional): The number of columns to compute the mean embeddings for. Defaults to 3.

    Returns:
    - col_embeddings_1 (torch.Tensor): The mean embeddings of the columns computed from the first text encoder.
    - col_embeddings_2 (torch.Tensor): The mean embeddings of the columns computed from the second text encoder.
    """
    with open(celeb_path, 'r') as f:
        celeb_names=f.read().splitlines()
    ''' get embeddings '''
    col_embeddings_1=[[]for _ in range(n_column)]
    col_embeddings_2=[[]for _ in range(n_column)]
    for name in tqdm(celeb_names,desc='get embeddings'):
        token_ids_1=tokenizer_1(
            name,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        token_ids_2=tokenizer_2(
            name,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0] # (max_length,)

        embeddings_1 = text_encoder_1.get_input_embeddings().weight.data[token_ids_1] # (n,768)
        embeddings_2 = text_encoder_2.get_input_embeddings().weight.data[token_ids_2] # (n,1280)

        for i in range(1,min(embeddings_1.shape[0]-1,n_column+1)):
            col_embeddings_1[i-1].append(embeddings_1[i].unsqueeze(0))
            col_embeddings_2[i-1].append(embeddings_2[i].unsqueeze(0))
    for i in range(n_column): 
        col_embeddings_1[i]=torch.cat(col_embeddings_1[i]).mean(dim=0).unsqueeze(0)
        col_embeddings_2[i]=torch.cat(col_embeddings_2[i]).mean(dim=0).unsqueeze(0)
    col_embeddings_1=torch.cat(col_embeddings_1) # (n,768)
    col_embeddings_2=torch.cat(col_embeddings_2) # (n,1280)
    
    return col_embeddings_1,col_embeddings_2

    
def image_grid(imgs, rows, cols):
    """
    Create a grid of images from a list of images.

    Args:
    - imgs (List[PIL.Image]): A list of images to create the grid from.
    - rows (int): The number of rows in the grid.
    - cols (int): The number of columns in the grid.

    Returns:
    - grid (PIL.Image): The resulting grid of images.
    """
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def setup_seed(seed):
    """
    Set the seed for the random number generators of torch, numpy, and random.

    Args:
        seed (int): The seed to set.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

@torch.no_grad()
def get_celeb_names_encoder_output(
    celeb_path : str,
    tokenizer : CLIPTokenizer,
    tokenizer_2 : CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    n_column: int=3,
):
    """
    Compute the mean embeddings of the columns in the celeb dataset and return the output hidden states of the text encoder.

    Args:
    - celeb_path (str): The path to the celeb dataset.
    - tokenizer (CLIPTokenizer): The tokenizer for the first text encoder.
    - tokenizer_2 (CLIPTokenizer): The tokenizer for the second text encoder.
    - text_encoder (CLIPTextModel): The first text encoder.
    - text_encoder_2 (CLIPTextModelWithProjection): The second text encoder.
    - n_column (int, optional): The number of columns to compute the mean embeddings for. Defaults to 3.

    Returns:
    - celeb_name_encoder_output (torch.Tensor): The output hidden states of the first text encoder.
    - celeb_name_encoder_output_2 (torch.Tensor): The output hidden states of the second text encoder.
    """
    celeb_name_embeds_mean, celeb_name_embeds_mean_2=get_celeb_col_mean(
        celeb_path=celeb_path,
        tokenizer_1=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder,
        text_encoder_2=text_encoder_2,
        n_column=n_column
    )
    token_embeds=text_encoder.get_input_embeddings().weight.data
    token_embeds_2=text_encoder_2.get_input_embeddings().weight.data
    celeb_name_embeds_mean=torch.cat([
        token_embeds[tokenizer.convert_tokens_to_ids(tokenizer.bos_token)].unsqueeze(0),
        celeb_name_embeds_mean,
        token_embeds[tokenizer.convert_tokens_to_ids(tokenizer.eos_token)].unsqueeze(0),
        token_embeds[tokenizer.convert_tokens_to_ids(tokenizer.pad_token)].unsqueeze(0).repeat(tokenizer.model_max_length-n_column-2,1)
    ],dim=0).unsqueeze(0)
    celeb_name_embeds_mean_2=torch.cat([
        token_embeds_2[tokenizer_2.convert_tokens_to_ids(tokenizer_2.bos_token)].unsqueeze(0),
        celeb_name_embeds_mean_2,
        token_embeds_2[tokenizer_2.convert_tokens_to_ids(tokenizer_2.eos_token)].unsqueeze(0),
        token_embeds_2[tokenizer_2.convert_tokens_to_ids(tokenizer_2.pad_token)].unsqueeze(0).repeat(tokenizer_2.model_max_length-n_column-2,1)
    ],dim=0).unsqueeze(0)
    
    celeb_name_encoder_output=text_encoder(
        input_ids=None,
        inputs_embeds=celeb_name_embeds_mean,
        output_hidden_states=True,
    ).hidden_states[-2][0,1:n_column+1]
    
    celeb_name_encoder_output_2=text_encoder_2(
        input_ids=None,
        inputs_embeds=celeb_name_embeds_mean_2,
        output_hidden_states=True,
    ).hidden_states[-2][0,1:n_column+1]
    
    return celeb_name_encoder_output,celeb_name_encoder_output_2

def find_files_recursively(dir_path, exts=['.jpg', '.png']):
    """
    Recursively find all files in a given directory path with specified extensions.

    Args:
        dir_path (str): The root directory path to search for files.
        exts (list[str], optional): A list of file extensions to search for. Defaults to ['.jpg', '.png'].

    Returns:
        list[str]: A list of file paths found.
    """
    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in exts):
                files.append(os.path.join(root, filename))
    return files

def write_dataset_to_file(dataset_list:list[str], save_path:str):
    """
    Write a list of dataset paths to a file, where each line in the file
    corresponds to a file in the dataset.
    
    Args:
        dataset_list (list[str]): A list of dataset paths.
        save_path (str): The path to save the file.
    """
    with open(save_path,'w') as f:
        for dataset in dataset_list:
            files=find_files_recursively(dataset)
            for file in files:
                f.write(file+'\n')