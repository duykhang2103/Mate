import torch
from tasks.eval.model_utils import load_llava_ov

print("Testing LLaVA-OneVision loading...")
model, processor = load_llava_ov(
    "MODELS/llava-onevision-7b",
    num_frames=16,
)
print(f"Model loaded successfully: {type(model)}")
print(f"Processor loaded successfully: {type(processor)}")

# Test basic inference
print("\nTesting basic inference...")
from PIL import Image
import requests

# Create a simple test image
test_image = Image.new('RGB', (224, 224), color='red')

# Prepare inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is this image?"},
        ],
    },
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[test_image], return_tensors="pt")
inputs = inputs.to(model.device, torch.bfloat16)

# Generate
print("Generating response...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

response = processor.decode(output[0], skip_special_tokens=True)
print(f"Response: {response}")

print("\nSmoke test completed successfully!")
