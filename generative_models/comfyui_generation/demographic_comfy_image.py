import json
import requests
import time
from io import BytesIO
from PIL import Image
import random

# ComfyUI server details
server_address = '127.0.0.1:8188'
client_id = str(time.time())  # Unique client ID

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = requests.post(f"http://{server_address}/prompt", data=data)
    print("API Response:", req.text)  # Debug: Print response for errors
    try:
        return req.json()
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from server: " + req.text)

def get_history(prompt_id):
    with requests.get(f"http://{server_address}/history/{prompt_id}") as response:
        return response.json()

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = requests.compat.urlencode(data)
    with requests.get(f"http://{server_address}/view?{url_values}") as response:
        return response.content

# Ethnicity, age, and gender prompts
descriptions = [
    "Portrait of a 3-year-old East Asian boy, short black hair, smiling with wide eyes, wearing a bright yellow t-shirt, smooth fair skin, natural outdoor lighting, playful mood.",
    "Teenage Middle Eastern girl, about 14, olive skin, long dark wavy hair, thoughtful expression, subtle gold earrings, wearing a turquoise dress, plain background, soft lighting.",
    "Photorealistic close-up of a 25-year-old Black man, deep brown skin, sharp jawline, short curly hair, trimmed beard, athletic build, casual white shirt, city background, energetic vibe.",
    "Portrait of a 40-year-old South Asian woman, medium-brown skin, elegant updo with wispy hair, soft smile, wearing a traditional sari, gold necklace, warm indoor light, gentle gaze.",
    "Highly detailed image of an elderly Hispanic man, about 75 years old, tan wrinkled skin, gray hair, mustache, wearing a straw hat and plaid shirt, rustic village background, kind expression.",
    "Pre-teen Caucasian girl, around 11 years old, freckles, wavy red hair, blue eyes, wearing a green hoodie, vivid smile, schoolyard background, daylight.",
    "Close-up of an 85-year-old Native American woman, long silver braided hair, high cheekbones, deep wrinkles, traditional patterned shawl, neutral background, dignified and wise mood.",
    "Portrait of a 35-year-old Southeast Asian man, tan skin, short straight dark hair, subtle stubble, brown eyes, light blue casual shirt, indoor background with soft ambient light, relaxed expression."
]

# Base workflow JSON template (corrected checkpoint name)
base_prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 0,
      "steps": 20,
      "cfg": 8.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["4", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    },
    "class_type": "KSampler"
  },
  "4": {
    "inputs": {
      "ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  },
  "5": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage"
  },
  "6": {
    "inputs": {
      "text": "DESCRIPTION_PLACEHOLDER",
      "clip": ["4", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "inputs": {
      "text": "blurry, low quality",
      "clip": ["4", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "8": {
    "inputs": {
      "samples": ["3", 0],
      "vae": ["4", 2]
    },
    "class_type": "VAEDecode"
  },
  "9": {
    "inputs": {
      "filename_prefix": "generated_image",
      "images": ["8", 0]
    },
    "class_type": "SaveImage"
  }
}
"""

# Generate images for each description
for i, desc in enumerate(descriptions, start=1):
    print(f"Generating image for description {i}: {desc}")

    # Load and customize the workflow
    workflow_str = base_prompt_text.replace("DESCRIPTION_PLACEHOLDER", desc)
    workflow = json.loads(workflow_str)

    # Randomize seed for variety
    workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

    # Queue the prompt
    response = queue_prompt(workflow)
    if 'prompt_id' not in response:
        print(f"Error for description {i}: Missing 'prompt_id' in response: {response}")
        continue

    prompt_id = response['prompt_id']
    print(f"Queued prompt ID: {prompt_id}")

    # Wait for completion
    while True:
        time.sleep(1)
        history = get_history(prompt_id)
        if prompt_id in history:
            break

    # Extract and save the image
    output = history[prompt_id]['outputs']
    image_data = list(output.values())[0]['images'][0]
    image_bytes = get_image(image_data['filename'], image_data['subfolder'], image_data['type'])
    image = Image.open(BytesIO(image_bytes))
    filename = f"generated_image_{i}.png"
    image.save(filename)
    print(f"Image saved as {filename}")