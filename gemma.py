# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an X-ray security inspection assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/home/hgyeo/Desktop/Colormapping/data/input_color/3.png"},
            {"type": "text", "text": "Please analyze the X-ray image and determine whether it contains any dangerous or prohibited items. If any are present, briefly state what the item is and where it is located (e.g., 'center', 'bottom right'). If nothing is found, simply say 'No threats detected.'"}
        ]
    }
]


inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
