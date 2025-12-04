from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import torchvision.io as io
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import cv2
import numpy as np
from torchvision.transforms import v2 as transforms
import argparse
from transformers import CLIPVisionModelWithProjection
import json

from modules.my_clip import CLIPTextModel as MyCLIPTextModel
from modules.my_clip import CLIPTextModelWithProjection as MyCLIPTextModelWithProjection
from modules.attention_processor import IPAttnProcessor2_0
from modules.uniid import UniID,load_face_recogniztion_model
from utils import image_grid, get_celeb_names_encoder_output
from modules.my_pipeline import MyStableDiffusionXLPipeline


class Inference:
    def __init__(
        self,
        pretrained_model_name_or_path: str | Path,
        pretrained_face_recog_model_path: str,
        insightface_root: str,
        num_text_branch_tokens: int,
        celeb_names_file: str = None,
        drop_text_branch: bool = False,
        load_text_branch_from_checkpoint:str=None,
        num_adapter_tokens: int = 16,
        normalize_adapter_output: bool = False,
        adapter_hidden_dim: int = 1024,
        drop_adapter: bool = False,
        load_adapter_branch_from_checkpoint: str | None = None,
        torch_dtype=torch.float16,
        device="cpu",
    ) -> None:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        
        text_encoder = MyCLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        ).to(device)
        text_encoder_2 = MyCLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        ).to(device)

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype
        ).to(device)

        self.face_recog_model = load_face_recogniztion_model(
            pretrained_face_recog_model_path=pretrained_face_recog_model_path,
            freeze_model=True,
        ).to(device=device, dtype=torch_dtype)
        self.model = UniID(
            unet=unet,
            text_encoder_hidden_size=text_encoder.config.hidden_size,
            text_encoder_2_hidden_size=text_encoder_2.config.hidden_size,
            drop_text_branch=drop_text_branch,
            load_text_branch_from_checkpoint=load_text_branch_from_checkpoint,
            num_text_branch_tokens=num_text_branch_tokens,
            num_adapter_tokens=num_adapter_tokens,
            adapter_hidden_dim=adapter_hidden_dim,
            normalize_adapter_output=normalize_adapter_output,
            drop_adapter=drop_adapter,
            load_adapter_branch_from_checkpoint=load_adapter_branch_from_checkpoint,
        ).to(device=device, dtype=torch_dtype)

        pipe = MyStableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
            unet=unet,
            safety_checker=None,
            torch_dtype=torch_dtype,
        ).to(device)
        self.num_text_branch_tokens = num_text_branch_tokens
        self.pipe = pipe
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.normalize_adapter_output = normalize_adapter_output
        self.drop_text_branch = drop_text_branch
        self.drop_adapter = drop_adapter
        self.celeb_names_file = celeb_names_file

        self.image_processor = CLIPImageProcessor()
        self.app = FaceAnalysis(
            name="buffalo_l", root=insightface_root, providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.inference_mode()
    def get_model_output(self, processed_image):
        face_recog_model_outputs, face_recog_model_hidden_states = self.face_recog_model(processed_image)

        text_branch_output, adapter_branch_output = self.model(
            face_recog_model_outputs, face_recog_model_hidden_states
        )
        (text_branch_output_1, text_branch_output_2) = text_branch_output

        uncond_text_branch_output, uncond_adapter_branch_output = self.model(
            torch.zeros_like(face_recog_model_outputs),
            [
                torch.zeros_like(face_recog_model_hidden_state)
                for face_recog_model_hidden_state in face_recog_model_hidden_states
            ],
        )
        (uncond_text_branch_output_1, uncond_text_branch_output_2) = uncond_text_branch_output
        return (
                (text_branch_output_1, text_branch_output_2, adapter_branch_output),
                (
                    uncond_text_branch_output_1,
                    uncond_text_branch_output_2,
                    uncond_adapter_branch_output,
                ),
        )

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor2_0):
                attn_processor.scale = scale

    def process_image(
        self,
        cv2_image: np.ndarray,
    ):
        faces = self.app.get(cv2_image)
        gender = 'man' if faces[0].gender==1 else 'woman'
        face_image = cv2.cvtColor(
            face_align.norm_crop(cv2_image, landmark=faces[0].kps, image_size=112),
            cv2.COLOR_BGR2RGB,
        )
        face_image = Image.fromarray(face_image)

        face_recog_image = (
            self.transform(face_image)
            .to(self.device, dtype=self.torch_dtype)
            .unsqueeze(0)
        )

        return face_recog_image, gender

    @torch.inference_mode()
    def infer(
        self,
        prompts: list[str],
        cv2_image: np.ndarray | None = None,
        num_images_per_prompt: int = 4,
        seed=None,
        prompt_guidance_scale=7.5,
        adapter_branch_scale=1.0,
        num_inference_steps=30,
        save_dir: str | Path | None = None,
        save_grid_images: bool = True,
        text_branch_scale: float = 1.0,
        infer_batch_size: int = 4,
        start_merge_step:int = 0,
        adapter_branch_scale_before_merge:float=0.0,
        text_branch_scale_before_merge:float=0.0,
        **kwargs,
    ):
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        face_recog_image, gender = self.process_image(cv2_image)

        (
            (text_branch_output_1, text_branch_output_2, adapter_branch_output),
            (
                _,
                _,
                uncond_adapter_branch_output,
            ),
        ) = self.get_model_output(face_recog_image)

        if not self.drop_adapter:
            bs_embed, seq_len, _ = adapter_branch_output.shape
            adapter_branch_output = adapter_branch_output.repeat(1, num_images_per_prompt, 1)
            adapter_branch_output = adapter_branch_output.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            uncond_adapter_branch_output = uncond_adapter_branch_output.repeat(
                1, num_images_per_prompt, 1
            )
            uncond_adapter_branch_output = uncond_adapter_branch_output.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

        if not self.drop_text_branch:
            if self.celeb_names_file is not None:
                (
                    celeb_name_encoder_output,
                    celeb_name_encoder_output_2,
                ) = get_celeb_names_encoder_output(
                    celeb_path=self.celeb_names_file,
                    tokenizer=self.pipe.tokenizer,
                    tokenizer_2=self.pipe.tokenizer_2,
                    text_encoder=self.pipe.text_encoder,
                    text_encoder_2=self.pipe.text_encoder_2,
                    n_column=self.num_text_branch_tokens,
                )
                celeb_name_encoder_output_norm = torch.norm(
                    celeb_name_encoder_output, dim=1, keepdim=True
                )
                celeb_name_encoder_output_norm_2 = torch.norm(
                    celeb_name_encoder_output_2, dim=1, keepdim=True
                )

                text_branch_output_1 = (
                    text_branch_output_1
                    / torch.norm(text_branch_output_1, dim=-1, keepdim=True)
                    * celeb_name_encoder_output_norm
                )
                text_branch_output_2 = (
                    text_branch_output_2
                    / torch.norm(text_branch_output_2, dim=-1, keepdim=True)
                    * celeb_name_encoder_output_norm_2
                )
            text_branch_output = torch.cat(
                [text_branch_output_1, text_branch_output_2], dim=-1
            )

        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        for prompt in prompts:
            torch.cuda.empty_cache()
            prompt = prompt.replace("person", gender)

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            if not self.drop_text_branch:
                prompt_embeds = torch.cat(
                    [
                        prompt_embeds[:, :1],
                        text_branch_output.expand(num_images_per_prompt, -1, -1),
                        prompt_embeds[:, 1 : -self.num_text_branch_tokens],
                    ],
                    dim=1,
                )

            if not self.drop_adapter:
                prompt_embeds = torch.cat([prompt_embeds, adapter_branch_output], dim=1)
                negative_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, uncond_adapter_branch_output], dim=1
                )
                
            if seed is not None:
                generator.manual_seed(seed)

            prompt_images = []
            for i in range(0, num_images_per_prompt, infer_batch_size):
                if i + infer_batch_size > num_images_per_prompt:
                    num_images = infer_batch_size - i
                else:
                    num_images = infer_batch_size

                if save_dir is not None:
                    save_dir = Path(save_dir)
                    save_dir.mkdir(exist_ok=True, parents=True)

                images = self.pipe(
                    prompt_embeds=prompt_embeds[i : i + num_images],
                    negative_prompt_embeds=negative_prompt_embeds[i : i + num_images],
                    pooled_prompt_embeds=pooled_prompt_embeds[i : i + num_images],
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[
                        i : i + num_images
                    ],
                    num_inference_steps=num_inference_steps,
                    generator=generator if seed is not None else None,
                    guidance_scale=prompt_guidance_scale,
                    start_merge_step=start_merge_step,
                    kwargs=kwargs,
                    adapter_branch_scale_before_merge=adapter_branch_scale_before_merge,
                    adapter_branch_scale=adapter_branch_scale,
                    num_text_branch_tokens=text_branch_output.shape[1],
                    text_branch_scale_before_merge=text_branch_scale_before_merge,
                    text_branch_scale=text_branch_scale,
                ).images
                prompt_images.extend(images)

                if save_dir is not None:
                    image_save_dir = save_dir.joinpath("_".join(prompt.split(" ")))
                    image_save_dir.mkdir(exist_ok=True)
                    for j, image in enumerate(images):
                        image.save(image_save_dir / f"{i+j}.jpg")
                    prompt_file = image_save_dir / "prompt.txt"
                    if not prompt_file.exists():
                        with open(prompt_file, "w") as f:
                            f.write(prompt)
            
            if save_grid_images:
                if num_images_per_prompt % 8 == 0:
                    grid_image = image_grid(
                        prompt_images, num_images_per_prompt // 8, 8
                    )
                else:
                    grid_image = image_grid(prompt_images, 1, num_images_per_prompt)
                grid_image.save(save_dir / (image_save_dir.name + ".jpg"))
        return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--celeb_names_file",
        type=str,
        default=None,
        help="Path to celeb names file",
    )
    parser.add_argument(
        "--insightface_root",
        type=str,
        default=None,
        required=True,
        help="Path to insightface models",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Prompt for the image generation.",
    )
    parser.add_argument(
        "--prompt_file", type=str, required=False, default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=42, required=False,
    )
    parser.add_argument(
        "--num_inference_steps", type=int, required=False, default=50,
    )
    parser.add_argument(
        "--num_images_per_prompt", required=False, default=16, type=int,
    )
    parser.add_argument(
        "--save_dir", required=False, default=None, type=str,
    )
    parser.add_argument("--load_adapter_from_checkpoint", required=False, type=str, default=None)
    parser.add_argument("--ref_images_dir", type=str, required=False, default=None)
    parser.add_argument(
        "--normalize_adapter_output",
        action="store_true",
        default=False,
        help="whether to use adaptive scale",
    )
    parser.add_argument(
        "--save_grid_images",
        action="store_true",
        default=False,
        help="whether to save grid images",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (like cuda:0 or cpu).",
    )
    parser.add_argument(
        "--text_branch_scale",
        type=float,
        default=None,
        help="scale of the text branch output for the reference images",
    )
    parser.add_argument(
        "--num_reference",
        type=int,
        default=20,
        help="number of reference images to use",
    )
    parser.add_argument(
        "--infer_batch_size", type=int, default=None, help="batch size of inference",
    )
    parser.add_argument(
        "--adapter_branch_scale", type=float, default=None, help="scale of the adapter branch",
    )
    parser.add_argument(
        "--load_text_branch_from_checkpoint",
        type=str,
        default=None,
        help="load text branch from checkpoint",
    )
    parser.add_argument(
        "--start_merge_step", 
        type=int, 
        default=0, 
        help="merge step",
    )
    parser.add_argument(
        "--adapter_branch_scale_before_merge", 
        type=float, 
        default=0.0, 
        help="scale of the ipa branch before merge",
    )
    parser.add_argument(
        "--text_branch_scale_before_merge", 
        type=float, 
        default=0.0, 
        help="scale of the text branch before merge",
    )
    parser.add_argument(
        "--ref_image_path", 
        type=str, 
        default=None, 
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("`--prompt` cannot be used with `--prompt_file`")

    if args.save_dir is not None:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.infer_batch_size is None:
        args.infer_batch_size = args.num_images_per_prompt
    else:
        args.infer_batch_size = min(args.infer_batch_size, args.num_images_per_prompt)

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.load_text_branch_from_checkpoint is not None:
        text_branch_config=json.load(open(f"{args.load_text_branch_from_checkpoint}/config.json",'r'))
    else:
        text_branch_config=None
    if args.load_adapter_from_checkpoint is not None:
        adapter_branch_config=json.load(open(f"{args.load_adapter_from_checkpoint}/config.json",'r'))
    else:
        adapter_branch_config=None
    torch_dtype = torch.float16

    inference = Inference(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_face_recog_model_path=text_branch_config['pretrained_face_recog_model_path'],
        insightface_root=args.insightface_root,
        num_text_branch_tokens=text_branch_config['num_text_branch_tokens'] if text_branch_config is not None else 3,
        celeb_names_file=args.celeb_names_file,
        drop_text_branch=True if args.load_text_branch_from_checkpoint is None else False,
        load_text_branch_from_checkpoint=args.load_text_branch_from_checkpoint+'/pytorch_model.bin',
        num_adapter_tokens=adapter_branch_config['num_adapter_tokens'] if adapter_branch_config is not None else 25,
        normalize_adapter_output=args.normalize_adapter_output,
        adapter_hidden_dim=adapter_branch_config['adapter_hidden_dim'] if adapter_branch_config is not None else 768,
        drop_adapter=True if args.load_adapter_from_checkpoint is None else False,
        load_adapter_branch_from_checkpoint=args.load_adapter_from_checkpoint+'/pytorch_model.bin',
        torch_dtype=torch.float16,
        device="cuda:0",
    )

    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file, "r") as f:
            prompts = f.read().splitlines()

    if args.ref_images_dir is not None:
        try:
            ref_images = sorted(
                Path(args.ref_images_dir).iterdir(), key=lambda x: int(x.name.split(".")[0])
            )
        except ValueError:
            ref_images = sorted(
                Path(args.ref_images_dir).iterdir(), key=lambda x: x.name.split(".")[0]
            )
    else:
        ref_images = [Path(args.ref_image_path)]

    print(args.ref_image_path)
    
    for i, image in enumerate(ref_images):
        if image.is_file() and i < args.num_reference:
            person_id = image.name.split(".")[0]
            save_dir = f"{args.save_dir}/{person_id}_seed_{args.seed}"
            # if Path(save_dir).exists():
            #     continue

            if image.name.endswith(".pt"):
                processed_image_dict = torch.load(image, weights_only=False)
                ref_image = None
            else:
                ref_image = cv2.imread(str(image))
                processed_image_dict = None

            inference.infer(
                prompts=prompts,
                cv2_image=ref_image,
                processed_image_dict=processed_image_dict,
                num_images_per_prompt=args.num_images_per_prompt,
                seed=args.seed,
                adapter_branch_scale=args.adapter_branch_scale,
                text_branch_scale=args.text_branch_scale,
                save_dir=save_dir,
                save_grid_images=args.save_grid_images,
                infer_batch_size=args.infer_batch_size,
                start_merge_step=args.start_merge_step,
                adapter_branch_scale_before_merge=args.adapter_branch_scale_before_merge,
                text_branch_scale_before_merge=args.text_branch_scale_before_merge,
            )
            torch.cuda.empty_cache()
