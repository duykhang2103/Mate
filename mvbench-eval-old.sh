#!/bin/bash

set -e  # Exit immediately if a command fails

echo "Installing PyTorch..."
# pip3 install torch torchvision

echo "Installing requirements..."
# pip install -r requirements.txt

echo "Installing additional packages..."
pip install urllib3 huggingface_hub

echo "Downloading model..."
# huggingface-cli download ermu2001/pllava-7b \
#     --local-dir MODELS/pllava-7b

hf download llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --local-dir MODELS/llava-onevision-7b

pip install "huggingface-hub==0.21.4"
pip install "transformers==4.46.3"

echo "Running inference..."
# python scripts/infer_single_video.py \
#     --video example/cooking.mp4 \
#     --model_dir MODELS/llava-onevision-7b \
#     --weight_dir MODELS/llava-onevision-7b \
#     --use_lora \
#     --lora_alpha 14 \
#     --question "What is happening in this video?" \
#     --num_frames 16 \
#     --pooling_shape 16-8-8
    # > loglog.log 2>&1

# echo "Done!"
# pip install -U huggingface_hub

# hf download OpenGVLab/MVBench \
#     --repo-type dataset \
#     --local-dir DATAS/MVBench
# unzip DATAS/MVBench.zip -d DATAS/MVBench/video

wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/star.zip
wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/sta.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/FunQA_test.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/clevrer.zip

cd DATAS/MVBench/video && for z in *.zip; do python -m zipfile -e "$z" .; done
cd ..
cd ..
cd ..

python -m tasks.eval.mvbench.ov_eval_mvbench \
    --pretrained_model_name_or_path MODELS/llava-onevision-7b \
    --save_path results/mvbench_ov_baseline_all \
    --num_frames 16 \
    --selected_layer 10 \
    --max_new_tokens 100 \
    --use_lora \
    --lora_alpha 14 \
    --top_p 1.0 \
    --temperature 1.0 \
    --pooling_shape 16-12-12 \
    --conv_mode eval_mvbench \
    --selected_layer 10 \
    --alpha 0.4 \
    --tau 0.8 \
    --temporal_segment_ratio 0.25 \
    --cluster_ratio 0.5 \
    --tasks "Action Sequence,Moving Direction" --max_samples 20


    #  --use_vtp --selected_layer 10 --alpha 0.4
    #  --tasks "Action Sequence,Moving Direction" --max_samples 20


    # > log-mvbench.log 2>&1
    # --tasks "Short Video" \
    # > log_mvbench_ov_baseline_all.log 2>&1



# hf download lmms-lab/Video-MME \
#     --repo-type dataset \
#     --local-dir DATAS/Video-MME

# pip install "transformers==4.46.3"

# cd DATAS/Video-MME && for f in videos_chunked_*.zip; do   unzip "$f"; done

# cd DATAS/Video-MME && for f in $(ls videos_chunked_*.zip | head -n 5); do    unzip "$f"; done




