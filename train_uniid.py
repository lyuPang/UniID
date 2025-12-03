import os
import argparse
from pathlib import Path
import json
import itertools
import time
import shutil
import math

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    CLIPTokenizer,
)

from modules.my_clip import CLIPTextModel as MyCLIPTextModel
from modules.my_clip import CLIPTextModelWithProjection as MyCLIPTextModelWithProjection
from modules.uniid import UniID,load_face_recogniztion_model
from utils import setup_seed,get_celeb_names_encoder_output
from load_data import collate_fn, UniIDDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--pretrained_face_recog_model_path",
        type=str,
        default=None,
        help="Path to pretrained resnet model. If not specified weights are initialized from the pretrained version in `torchvision.models`.",
    )
    parser.add_argument(
        "--insightface_root",
        type=str,
        default='~/.insightface',
        help="Path to insightface models",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=False,
        help="Training data root path",
    )
    parser.add_argument(
        "--data_path_file",
        type=str,
        default=None,
        required=False,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="uniid_output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help=("The resolution for input images"),
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument("--num_train_epochs", type=int, default=-1)
    parser.add_argument("--num_train_steps", type=int, default=-1)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_epoch",
        type=int,
        default=-1,
        help=("Save a checkpoint of the training state every X epoch"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_text_branch_tokens", type=int, default=2)
    parser.add_argument("--num_adapter_tokens", type=int, default=4)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--tracker_project", type=str, default="test_project", required=False)
    parser.add_argument("--tracker_name", type=str, default="test_name", required=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--image_drop_rate",
        type=float,
        default=0.05,
        help="image drop rate",
    )
    parser.add_argument(
        "--text_drop_rate",
        type=float,
        default=0.05,
        help="text drop rate",
    )
    parser.add_argument(
        "--load_text_branch_from_checkpoint",
        type=str,
        default=None,
        help="directory to the checkpoint of text branch",
    )
    parser.add_argument(
        "--freeze_text_branch",
        action="store_true",
        default=False,
        help="whether to freeze text branch",
    )
    parser.add_argument(
        "--normalize_adapter_output",
        action="store_true",
        default=False,
        help="whether to use normalize adapter output",
    )
    parser.add_argument(
        '--adapter_hidden_dim',
        type=int,
        default=768,
        help='hidden dim of adapter'
    )
    parser.add_argument(
        '--drop_text_branch',
        action="store_true",
        default=False,
        help="whether to drop text branch",
    )
    parser.add_argument(
        '--drop_adapter',
        action="store_true",
        default=False,
        help="whether to drop adapter",
    )
    parser.add_argument(
        '--freeze_adapter',
        action="store_true",
        default=False,
        help="whether to freeze adapter",
    )
    parser.add_argument(
        '--celeb_names_file',
        type=str,
        default=None,
        help="celeb name file",
    )
    parser.add_argument(
        '--text_branch_output_scale',
        type=float,
        default=1.0,
        help="scale factor of text branch output",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    args.output_dir = str(output_dir)

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name=args.tracker_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.tracker_name}} if args.report_to == "wandb" else {},
    )

    set_seed(args.seed)
    setup_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = MyCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_2 = MyCLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    face_recognition_model = load_face_recogniztion_model(
        pretrained_face_recog_model_path=args.pretrained_face_recog_model_path,
        freeze_model=True
    )
    
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    if not args.drop_text_branch and args.load_text_branch_from_checkpoint is not None:
        text_branch_config=json.load(open(f'{args.load_text_branch_from_checkpoint}/config.json','r'))
        text_branch_info={
            'num_text_branch_tokens':text_branch_config['num_text_branch_tokens'],
            'load_text_branch_from_checkpoint':f'{args.load_text_branch_from_checkpoint}/pytorch_model.bin',
        }
    else:
        text_branch_info={
            'num_text_branch_tokens':args.num_text_branch_tokens,
        }

    model = UniID(
        unet=unet,
        **text_branch_info,
        drop_text_branch=args.drop_text_branch,
        num_adapter_tokens=args.num_adapter_tokens,
        adapter_hidden_dim=args.adapter_hidden_dim,
        normalize_adapter_output=args.normalize_adapter_output,
        drop_adapter=args.drop_adapter,
    )
    if args.freeze_text_branch:
        model.freeze_text_branch()
    if args.freeze_adapter:
        model.freeze_adapter()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)  # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    face_recognition_model.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(
        [params for params in model.parameters() if params.requires_grad],
    )
    optimizer = torch.optim.AdamW(
        params_to_opt, 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )

    # celebrity name
    if not args.drop_text_branch and args.celeb_names_file is not None:
        celeb_name_encoder_output,celeb_name_encoder_output_2=get_celeb_names_encoder_output(
            celeb_path=args.celeb_names_file,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            n_column=args.num_text_branch_tokens
        )
        celeb_name_encoder_output_norm=torch.norm(celeb_name_encoder_output, dim=1,keepdim=True)
        celeb_name_encoder_output_norm_2=torch.norm(celeb_name_encoder_output_2, dim=1,keepdim=True)
        
    train_dataset = UniIDDataset(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        data_dir=args.data_root_path,
        data_path_file=args.data_path_file,
        i_drop_rate=args.image_drop_rate,
        t_drop_rate=args.text_drop_rate,
        insightface_root=args.insightface_root,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    skipped_epoch=0
    global_step = 0
    steps_per_epoch = math.ceil(len(train_dataset) / (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps))
    if args.load_from_checkpoint is not None:
        accelerator.load_state(args.load_from_checkpoint)
        skipped_epoch=int(args.load_from_checkpoint.split('-')[-1])
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
    
    num_train_epochs=args.num_train_epochs
    if num_train_epochs<0:
        num_train_epochs=args.num_train_steps//steps_per_epoch+1

    accelerator.print("Begin training...")
    
    for epoch in range(skipped_epoch, num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(model):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=torch.float32)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1)
                    ).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    face_recog_model_outputs, face_recog_model_hidden_states=face_recognition_model(batch["recog_model_input"].to(accelerator.device, dtype=weight_dtype))
                
                text_branch_output, adapter_branch_output = model(
                    face_recog_model_outputs=face_recog_model_outputs,
                    face_recog_model_hidden_states=face_recog_model_hidden_states,
                    text_branch_masks=batch["drop_text_embeds"].to(accelerator.device),
                    adapter_branch_masks=batch["drop_image_embeds"].to(accelerator.device)
                )

                text_input_ids, text_input_ids_2 = (
                    batch["text_input_ids"].to(accelerator.device),
                    batch["text_input_ids_2"].to(accelerator.device),
                )
                token_embeddings, token_embeddings_2 = (
                    text_encoder.get_input_embeddings().weight.data,
                    text_encoder_2.get_input_embeddings().weight.data,
                )
                text_embeddings, text_embeddings_2 = (
                    token_embeddings[text_input_ids[0]],
                    token_embeddings_2[text_input_ids_2[0]],
                )
                text_embeddings, text_embeddings_2 = (
                    text_embeddings.repeat((bsz, 1, 1)),
                    text_embeddings_2.repeat((bsz, 1, 1)),
                )

                encoder_output = text_encoder(
                    input_ids=text_input_ids,
                    inputs_embeds=text_embeddings,
                    output_hidden_states=True,
                )
                text_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(
                    input_ids=text_input_ids_2,
                    inputs_embeds=text_embeddings_2,
                    output_hidden_states=True,
                )
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                text_embeds = torch.concat(
                    [text_embeds, text_embeds_2], dim=-1
                )  # concat
                
                # replace text encoder output with ti_resampler output
                if not args.drop_text_branch:
                    if args.celeb_names_file is not None:
                        (text_branch_output_1, text_branch_output_2) = text_branch_output
                        text_branch_output_1 = text_branch_output_1/torch.norm(text_branch_output_1, dim=-1, keepdim=True)*celeb_name_encoder_output_norm # (bs, n_token, channel)
                        text_branch_output_2 = text_branch_output_2/torch.norm(text_branch_output_2, dim=-1, keepdim=True)*celeb_name_encoder_output_norm_2
                        text_branch_output = torch.concat([text_branch_output_1, text_branch_output_2], dim=-1)
                    else:
                        text_branch_output=torch.cat(text_branch_output,dim=-1)
                    text_branch_output=text_branch_output*args.text_branch_output_scale
                    text_embeds=torch.cat([text_embeds[:,:1],text_branch_output,text_embeds[:,1:-args.num_text_branch_tokens]],dim=1)
                    
                if not args.drop_adapter:
                    encoder_hidden_state = torch.cat([text_embeds,adapter_branch_output],dim=1)
                else:
                    encoder_hidden_state = text_embeds

                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(
                    accelerator.device, dtype=weight_dtype
                )
                unet_added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                }

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_state,
                    added_cond_kwargs=unet_added_cond_kwargs,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = (
                    accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                )

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                accelerator.log({
                    "step_loss": avg_loss,
                })

                if accelerator.is_main_process:
                    print(
                        "Epoch {}, step {}, global_step {}, data_time: {}, total_time: {}, step_loss: {}, lr: {}".format(
                            epoch,
                            step,
                            global_step,
                            load_data_time,
                            time.perf_counter() - begin,
                            avg_loss,
                            optimizer.param_groups[0]["lr"],
                        )
                    )

            global_step += 1
            begin = time.perf_counter()
            
            if args.num_train_steps>0 and global_step>args.num_train_steps:
                break
           
        if args.save_epoch>0 and (epoch + 1) % args.save_epoch == 0 and accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
            if args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[:num_to_remove]

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(
                            args.output_dir, removing_checkpoint
                        )
                        shutil.rmtree(removing_checkpoint)
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            accelerator.save_state(save_path, safe_serialization=False, timeout=60)
            with open(os.path.join(save_path,'config.json'),'w') as f:
                json.dump(vars(args),f,indent=2)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
