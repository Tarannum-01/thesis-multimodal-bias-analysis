import os  # <-- This was missing, now added
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch

# Load your fine-tuned model
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "./finetuned_vlm_gpu")

# The 3 images you asked for
image_paths = [
    "./fine tune/rotated_90_rotated_180_1 (1000).jpg",
    "./fine tune/rotated_180_rotated_90_1000_F.jpg",
    "./fine tune/rotated_90_rotated_180_1 (1246).jpg"
]

prompt = "<image>\nDescribe this image in detail, including any people, faces, age, ethnicity, gender, and emotional expression."

print("Starting inference on 3 images...\n")

for i, path in enumerate(image_paths, 1):
    print(f"IMAGE {i}/3: {os.path.basename(path)}")
    print("-" * 70)
    
    image = Image.open(path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)
    
    description = processor.decode(output[0], skip_special_tokens=True)
    print(description)
    print("\n" + "="*80 + "\n")

print("All 3 images tested successfully!")