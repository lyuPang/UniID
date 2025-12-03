model_name="stabilityai/stable-diffusion-xl-base-1.0"

text_branch_checkpoint_dir="path to the text branch checkpoint"
adapter_branch_checkpoint_dir="path to the adapter branch checkpoint"

ref_image_path="path to the reference image"
save_dir="path to the output directory"
prompt="input prompt" # or you can use `--prompt_file`

seed=42
start_merge_step=5
text_branch_scale_before_merge=0.5
text_branch_scale_after_merge=1.8
adapter_branch_scale_before_merge=0.8
adapter_branch_scale_after_merge=1.2

python inference.py \
      --pretrained_model_name_or_path $model_name \
      --load_text_branch_from_checkpoint $text_branch_checkpoint_dir \
      --load_adapter_from_checkpoint $adapter_branch_checkpoint_dir \
      --ref_image_path $ref_image_path \
      --prompt "${prompt}" \
      --seed $seed \
      --num_inference_steps 30 \
      --num_images_per_prompt 16 \
      --infer_batch_size 16 \
      --adapter_branch_scale $adapter_branch_scale_after_merge \
      --celeb_names_file "dataset/wiki_names_v2.txt" \
      --normalize_adapter_output \
      --text_branch_scale $text_branch_scale_after_merge \
      --start_merge_step $start_merge_step \
      --adapter_branch_scale_before_merge $adapter_branch_scale_before_merge \
      --text_branch_scale_before_merge $text_branch_scale_before_merge \
      --save_dir "${save_dir}" \
      --save_grid_images