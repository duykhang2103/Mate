#!/bin/bash

set -e  # Exit immediately if a command fails

echo "Installing PyTorch..."
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    # --index-url https://download.pytorch.org/whl/cu124

echo "Installing requirements..."
# pip install -r requirements.txt

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