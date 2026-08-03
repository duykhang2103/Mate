python -m tasks.eval.videomme.ov_eval_videomme \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/videomme_ov_baseline_fixed_short \
    --num_frames 16 \
    --selected_layer 10 \
    --max_new_tokens 100 \
    --tasks "Short Video" \
    > log_videomme_ov_baseline_fixed_short.log 2>&1


python -m tasks.eval.videomme.ov_eval_videomme \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/videomme_ov_vtp_alpha04_fixed_short \
    --num_frames 16 \
    --selected_layer 10 \
    --alpha 0.4 \
    --use_vtp \
    --max_new_tokens 100 \
    --tasks "Short Video" \
    > log_videomme_ov_vtp_alpha04_fixed_short.log 2>&1


python -m tasks.eval.videomme.ov_eval_videomme \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/videomme_ov_baseline_fixed_medium \
    --num_frames 16 \
    --selected_layer 10 \
    --max_new_tokens 100 \
    --tasks "Medium Video" \
    > log_videomme_ov_baseline_fixed_medium.log 2>&1


python -m tasks.eval.videomme.ov_eval_videomme \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/videomme_ov_baseline_fixed_long \
    --num_frames 16 \
    --selected_layer 10 \
    --max_new_tokens 100 \
    --tasks "Long Video" \
    > log_videomme_ov_baseline_fixed_long.log 2>&1



