#!/bin/bash

set -e  # Exit immediately if a command fails

echo "Installing PyTorch..."
# pip3 install torch torchvision

echo "Installing requirements..."
# pip install -r requirements.txt

echo "Installing additional packages..."
pip install urllib3 huggingface_hub

echo "Downloading model..."

# if [ -z "$(ls -A MODELS/llava-onevision-7b 2>/dev/null)" ]; then
#     echo "⬇️ Đang tải xuống LLaVA OneVision-7B..."
#     hf download llava-hf/llava-onevision-qwen2-7b-ov-hf \
#     --local-dir MODELS/llava-onevision-7b
# else
#     echo "✅ LLaVA OneVision-7B đã tồn tại trong Volume. Bỏ qua tải xuống."
# fi

if [ -z "$(ls -A MODELS/pllava-7b 2>/dev/null)" ]; then
    echo "⬇️ Đang tải xuống PLLAVA-7B..."
    # hf download llava-hf/llava-onevision-qwen2-7b-ov-hf \
    # --local-dir MODELS/llava-onevision-7b
    hf download ermu2001/pllava-7b \
    --local-dir MODELS/pllava-7b
else
    echo "✅ PLLAVA-7B đã tồn tại trong Volume. Bỏ qua tải xuống."
fi

# hf download llava-hf/llava-onevision-qwen2-7b-ov-hf \
#     --local-dir MODELS/llava-onevision-7b

pip install "huggingface-hub==0.21.4"
pip install "transformers==4.46.3"

# python scripts/infer_single_video_ov.py --video example/cooking.mp4

# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/star.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/sta.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/FunQA_test.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/clevrer.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/Moments_in_Time_Raw.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/data0613.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/perception.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/scene_qa.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/ssv2_video.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/tvqa.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/vlnqa.zip

# if [ "$(find DATAS/MVBench/video -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)" -lt 5 ]; then 
#     URLS=(
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/star.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/sta.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/FunQA_test.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/clevrer.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/Moments_in_Time_Raw.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/data0613.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/scene_qa.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/ssv2_video.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/perception.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/tvqa.zip"
#     "https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/vlnqa.zip"
#     )

#     # 2. Loop through the array
#     # 2. Loop through the array
#     for i in "${!URLS[@]}"; do
#         # Thêm -nc để tránh tải lại những file đã có sẵn trong /dataset
#         wget -nc -q --show-progress -P DATAS/MVBench/video "${URLS[$i]}" &
        
#         # Check if we have reached a multiple of 4 (indexes: 3, 7, 11, 15, 19)
#         if (( (i + 1) % 4 == 0 )); then
#             echo "Batch $(( (i + 1) / 4 )) sent. Waiting for it to finish..."
#             wait
#         fi
#     done
#     wait

#     echo "All downloads completed successfully!"


#     # hf download OpenGVLab/MVBench \
#     #     --repo-type dataset \
#     #     --local-dir DATAS/MVBench
#     # python -m zipfile -e DATAS/MVBench.zip -d DATAS/MVBench/video

#     # mv DATAS/MVBench.zip DATAS/MVBench/video
#     echo "Extracting zip files in DATAS/MVBench/video..."
#     ls -l DATAS/MVBench/video

#     cd DATAS/MVBench/video && for z in *.zip; do python -m zipfile -e "$z" .; done



#     cd ..
#     cd ..
#     cd ..
# else
#     echo "✅ PLLAVA-7B đã tồn tại trong Volume. Bỏ qua tải xuống."
# fi

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
python -m tasks.eval.videomme.pllava_eval_videomme \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --save_path results/pllava_videomme_all_prunevid_optical_motion \
    --num_frames 16 --use_lora --lora_alpha 14 \
    --conv_mode eval_videomme \
    --alpha 0.4 --selected_layer 10 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --use_flow_pruning --flow_dynamic_ratio 0.5 \
    --use_motion_adaptive --motion_scale 1.0 \

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