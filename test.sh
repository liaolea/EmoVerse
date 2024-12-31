CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type internvl2-4b \
    --model_id_or_path emoverse-4b \
    --system "You are an sentiment and emotion analysis expert." \
    --val_dataset sentiment_test_non_neg.json \
    --max_length 2048
