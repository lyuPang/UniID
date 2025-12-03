base_model="stabilityai/stable-diffusion-xl-base-1.0"

pretrained_face_recog_model_path="path to the face recognition model"
data_path_file="path to the data file"
output_root="path to the output directory"

seed=42
num_text_branch_tokens=3
text_branch_num_train_epochs=12
text_branch_save_epoch=12
text_branch_lr="1e-4"


accelerate launch train_uniid.py \
  --pretrained_model_name_or_path $base_model \
  --pretrained_face_recog_model_path "${pretrained_face_recog_model_path}" \
  --learning_rate "${text_branch_lr}" \
  --num_text_branch_tokens $num_text_branch_tokens \
  --data_path_file $data_path_file \
  --output_dir "${output_root}/text_branch_checkpoints" \
  --train_batch_size 23 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 32 \
  --num_train_epochs $text_branch_num_train_epochs \
  --save_epoch $text_branch_save_epoch \
  --mixed_precision "no" \
  --seed $seed \
  --drop_adapter

adapter_branch_num_train_epochs=16
adapter_branch_save_epoch=16
adapter_branch_lr="1e-5"

accelerate launch train_uniid.py \
  --pretrained_model_name_or_path $base_model \
  --pretrained_face_recog_model_path "${pretrained_face_recog_model_path}" \
  --num_adapter_tokens 25 \
  --data_path_file $data_path_file \
  --output_dir "${output_root}/adapter_branch_checkpoints" \
  --learning_rate $adapter_branch_lr \
  --train_batch_size 19 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 32 \
  --num_train_epochs $adapter_branch_num_train_epochs \
  --save_epoch $adapter_branch_save_epoch \
  --mixed_precision "no" \
  --seed $seed \
  --drop_text_branch