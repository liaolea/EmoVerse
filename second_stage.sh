CUDA_VISIBLE_DEVICES=1 \
swift sft \
  --model_type internvl2-4b \
  --model_id_or_path first_model_ckpt \
  --dataset emt_data_second.json \
  --sft_type lora \
  --output_dir output \
  --system "You are an sentiment and emotion analysis expert." \
  --num_train_epochs 2 \
  --max_length 2048 \
  --gradient_checkpointing true \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_steps 1000 \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --acc_strategy sentence
