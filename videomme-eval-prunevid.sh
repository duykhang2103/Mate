#!/bin/bash

set -e  # Exit immediately if a command fails


echo "Installing additional packages..."
pip install urllib3 huggingface_hub

echo "Downloading model..."

if [ -z "$(ls -A MODELS/pllava-7b 2>/dev/null)" ]; then
    echo "⬇️ Đang tải xuống PLLAVA-7B..."
    hf download ermu2001/pllava-7b \
    --local-dir MODELS/pllava-7b
else
    echo "✅ PLLAVA-7B đã tồn tại trong Volume. Bỏ qua tải xuống."
fi

pip install "huggingface-hub==0.21.4"
pip install "transformers==4.46.3"


if [ -z "$(ls -A DATAS/Video-MME 2>/dev/null)" ]; then
    hf download lmms-lab/Video-MME \
        --repo-type dataset \
        --local-dir DATAS/Video-MME
    cd DATAS/Video-MME && for f in videos_chunked_*.zip; do  unzip "$f"; done
    # cd DATAS/Video-MME && for f in $(ls videos_chunked_*.zip | head -n 2); do    unzip "$f"; done

    cd ..
    cd ..
else
    echo "✅ Video-MME đã tồn tại trong Volume. Bỏ qua tải xuống."
fi



# python videomme-json-converter.py 

### Smoke Test
# python scripts/infer_single_video.py \
#     --video example/cooking.mp4 \
#     --model_dir MODELS/pllava-7b \
#     --weight_dir MODELS/pllava-7b \
#     --use_lora --lora_alpha 14 \
#     --selected_layer 10 --alpha 0.4 \
#     --use_flow_pruning --flow_dynamic_ratio 0.5 \
#     --use_motion_adaptive --motion_scale 1.0

### MVBench Evaluation :  Task All
# python -m tasks.eval.mvbench.pllava_eval_mvbench \
#     --pretrained_model_name_or_path MODELS/pllava-7b \
#     --save_path test_results/flow_merge_only \
#     --num_frames 16 --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b \
#     --pooling_shape 16-12-12 --selected_layer 999 --alpha 0.4 --tau 0.8 \
#     --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
#     --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" \
#     --use_flow_pruning --flow_dynamic_ratio 0.5 \
#     --conv_mode plain --max_samples 200 \
#     --use_motion_adaptive --motion_scale 1.0 \


### Videomme Evaluation :  Task Short
# python -m tasks.eval.videomme.pllava_eval_videomme \
#     --pretrained_model_name_or_path MODELS/pllava-7b \
#     --save_path results/pllava_videomme_smoke_baseline \
#     --num_frames 16 --use_lora --lora_alpha 14 \
#     --conv_mode eval_videomme \
#     --alpha 1.0 --selected_layer 10 \
#     --tasks short --max_samples 10 \
#     --use_flow_pruning --flow_dynamic_ratio 0.5 \
#     --use_motion_adaptive --motion_scale 1.0 \

#### Videomme Evaluation :  Task All
# python -m tasks.eval.videomme.pllava_eval_videomme \
#     --pretrained_model_name_or_path MODELS/pllava-7b \
#     --save_path results/pllava_videomme_all_prunevid_optical_motion \
#     --num_frames 16 --use_lora --lora_alpha 14 \
#     --conv_mode eval_videomme \
#     --alpha 0.4 --selected_layer 10 --tau 0.8 \
#     --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
#     --use_flow_pruning --flow_dynamic_ratio 0.5 \
#     --use_motion_adaptive --motion_scale 1.0 

# python -m tasks.eval.mvbench.ov_eval_mvbench \
#     --pretrained_model_name_or_path MODELS/llava-onevision-7b \
#     --save_path results/mvbench_ov_vtp_all \
#     --num_frames 16 \
#     --max_new_tokens 100 \
#     --top_p 1.0 \
#     --temperature 1.0 \
#     --conv_mode eval_mvbench \
#     --use_vtp --selected_layer 10 --alpha 0.4

# python -m tasks.eval.mvbench.ov_eval_mvbench \
# python -m tasks.eval.mvbench.pllava_eval_mvbench \
#     --pretrained_model_name_or_path MODELS/pllava-7b  \
#     --weight_dir MODELS/pllava-7b  \
#     --save_path results/mvbench_pllava_prunevid_all \
#     --num_frames 16 \
#     --use_lora --lora_alpha 14 \
#     --pooling_shape 16-12-12 \
#     --selected_layer 10 \
#     --alpha 0.4 \
#     --tau 0.8 \
#     --temporal_segment_ratio 0.25 \
#     --cluster_ratio 0.5 \
#     --conv_mode eval_mvbench \
#     --max_new_tokens 100 \
#     --use_flow_pruning --flow_dynamic_ratio 0.5 \
#     --use_motion_adaptive --motion_scale 1.0 \


# python -m pytest tests/test_query_guided_merge.py -q


### Evaluation: All - query guided merge 
# python -m tasks.eval.videomme.pllava_eval_videomme \
#     --pretrained_model_name_or_path MODELS/pllava-7b \
#     --save_path results/pllava_videomme_query_merge_w07_r075_full \
#     --num_frames 16 --use_lora --lora_alpha 14 \
#     --conv_mode eval_videomme \
#     --alpha 0.4 --selected_layer 10 --tau 0.8 \
#     --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
#     --use_query_guided_merge \
#     --query_merge_weight 0.7 \
#     --query_dynamic_ratio 0.75 


python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path results/pllava_videomme_novelty_w00_r075_full \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_query_guided_merge \
    --query_merge_weight 0.0 \
    --query_dynamic_ratio 0.75




python scripts/compare_paired_results.py \
    --baseline zzresults/Video-MME/results/pllava_videomme_query_merge_w07_r075_full/pllava_videomme_query_merge_w07_r075_full/all_results.json \
    --candidate results/pllava_videomme_novelty_w00_r075_full/all_results.json \
    --json-output results/pllava_videomme_novelty_w00_r075_full/paired_vs_query_metrics.json
