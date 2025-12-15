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

# The exact image you asked for
image_path = "./fine tune/rotated_180_rotated_90_1000_F.jpg"

image = Image.open(image_path).convert("RGB")
prompt = "<image>\nDescribe this image in detail, including any people, faces, age, ethnicity, gender, and emotional expression."

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)

result = processor.decode(output[0], skip_special_tokens=True)
print("\n" + result)