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

# List of image descriptions
descriptions = [
    "Portrait of a young man in his late teens or early twenties with striking, intense bright teal or blue-green eyes, angular sharp face, prominent nose, strong jawline, dark arched eyebrows, dark slightly messy hair falling across forehead, slightly pursed mouth, contemplative expression, wearing a chunky knitted pale muted yellow-green beanie and dark gray or black t-shirt, soft diffused lighting, tightly framed close-up.",
    "Close-up slightly elevated downward-looking portrait of a young woman with serious contemplative expression, fair skin with natural redness in cheeks, striking bright blue eyes with thick dark eyelashes, dark tousled slightly messy hair framing face, slightly parted lips, wearing chunky beige or light-khaki knit beanie and dark blue or charcoal-colored top, soft diffused lighting, solid muted gray or blue background, shallow depth of field.",
    "Striking black and white portrait of an elderly woman in her late 80s or 90s with muted thoughtful expression, deeply wrinkled textured skin, dark somewhat narrowed hooded eyes with dark circles, slightly hooked prominent nose, thin lips in loose bun dark thick slightly unruly hair, wearing dark knitted shawl or scarf, soft diffused lighting, wide tonal range.",
    "Striking somewhat surreal portrait of a young man lying on back with thoughtful expression eyes closed, dark short hair, wearing dark textured jacket with patch on sleeve, canine reflection of medium-sized fluffy tan and white dog with intense serious gaze superimposed over face, diffused even lighting, symmetrical composition, soft dreamlike atmosphere, blurred subtle gradient background.",
    "Candid shot of middle-aged man in profound sad sorrowful expression with downcast eyes and furrowed eyebrows, dark short hair, wearing dark heavy-duty jacket with fur-lined hood up, low angle looking up, soft diffused lighting, background of corral filled with sheep and pale blue overcast sky.",
    "High-contrast black and white portrait of elderly man with muted pensive expression, incredibly detailed deep wrinkled textured skin, dark somewhat closed eyes, large prominent nose with bumps, slightly downturned mouth, mix of short wiry and longer patches hair, wide bulbous nose and cheekbones, high contrast lighting with intense highlights and deep shadows, tight composition.",
    "Stark intensely detailed black and white portrait of very old proboscis monkey head, heavily wrinkled sagging skin with deep fissures and folds, immense bulbous swollen nose with maze of wrinkles, thin fragile skin translucent in places, thick dark tangled unruly hair partially obscuring upper face, prominent ear, flat even lighting with extreme tonal range, intensely tight composition.",
    "Collage of eleven diverse portraits representing various professions, ages, ethnicities: 1. Woman 25-35 in red and white pattern as fashion model; 2. Bearded man 35-50 as chef; 3. Woman 25-40 in white chef's hat as baker; 4. Man with braids 20-30 as stylist; 5. Man in black suit 35-55 as executive; 6. Man in white shirt dark tie 30-50 as teacher; 7. Man in grey suit 40-60 as railway worker; 8. Man in red shirt gold badge 30-50 as police officer; 9. Man in blue white striped shirt 40-60 as fisherman; 10. Man in yellow hard hat 30-50 as construction worker."
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