import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import cv2


class UniIDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        tokenizer_2,
        insightface_root,
        data_dir,
        data_path_file:str=None,
        size=512,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        ti_drop_rate=0.05,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate=ti_drop_rate
        self.app = FaceAnalysis(
            name="buffalo_l", root=insightface_root
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        
        self.data_path=[]
        if data_dir is None:
            for sub_dir in Path(data_dir).iterdir():
                self.data_path+=[f for f in sub_dir.iterdir() if f.is_file()]
        elif data_path_file is not None:
            with open(data_path_file,'r') as f:
                data_path=f.readlines()
            self.data_path=[path.strip() for path in data_path]
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        # text
        text = "a photo of a person"

        # read image
        image_path = self.data_path[idx]
        image = cv2.imread(image_path)

        # crop face
        try:
            face = self.app.get(image)[0]
            replace_token="man" if face.gender==1 else "woman"
            face_image = cv2.cvtColor(
                face_align.norm_crop(image, landmark=face.kps, image_size=112),
                cv2.COLOR_BGR2RGB,
            )
            face_image = Image.fromarray(face_image)
        except Exception as e:
            replace_token="person"
            face_image=Image.fromarray(image).resize((112,112))
            print(image_path)
            
        text=text.replace("person",replace_token)
        
        face_recog_image = self.transform(face_image)
        
        # original size
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        original_width, original_height = image.size
        original_size = torch.tensor([original_height, original_width])
        
        image=image.resize((self.size, self.size), resample=Image.BILINEAR)
        image = self.transform(image)

        # drop
        drop_image_embed = False
        drop_text_embed = False
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = True
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            drop_text_embed = True
        elif rand_num<(self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            drop_image_embed=True
            drop_text_embed=True

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "image": image,
            "recog_model_input": face_recog_image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "drop_image_embed": drop_image_embed,
            "drop_text_embed": drop_text_embed,
            "original_size": original_size,
            "crop_coords_top_left": torch.tensor([0, 0]),
            "target_size": torch.tensor([self.size, self.size]),
        }

    def __len__(self):
        return len(self.data_path)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    recog_model_input = torch.stack([example["recog_model_input"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    drop_text_embeds = torch.tensor([example["drop_text_embed"] for example in data], dtype=torch.bool)
    drop_image_embeds = torch.tensor([example["drop_image_embed"] for example in data], dtype=torch.bool)
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack(
        [example["crop_coords_top_left"] for example in data]
    )
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "recog_model_input": recog_model_input,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "drop_text_embeds": drop_text_embeds,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }