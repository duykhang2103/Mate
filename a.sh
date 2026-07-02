#!/bin/bash

set -e  # Exit immediately if a command fails

echo "Installing PyTorch..."
pip3 install torch torchvision

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing additional packages..."
pip install urllib3 huggingface_hub

echo "Downloading model..."
huggingface-cli download ermu2001/pllava-7b \
    --local-dir MODELS/pllava-7b

echo "Running inference..."
python scripts/infer_single_video.py \
    --video example/cooking.mp4 \
    --model_dir MODELS/pllava-7b \
    --weight_dir MODELS/pllava-7b \
    --use_lora \
    --lora_alpha 14 \
    --question "What is happening in this video?" \
    --num_frames 16 \
    --pooling_shape 16-8-8 \
    > loglog.log 2>&1

echo "Done!"

hf download lmms-lab/Video-MME \
    --repo-type dataset \
    --local-dir DATAS/Video-MME

# cd DATAS/Video-MME && for f in videos_chunked_*.zip; do   unzip "$f"; done

cd DATAS/Video-MME && for f in $(ls videos_chunked_*.zip | head -n 5); do    unzip "$f"; done




# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/star.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/sta.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/FunQA_test.zip
# wget -P DATAS/MVBench/video https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/main/video/clevrer.zip
