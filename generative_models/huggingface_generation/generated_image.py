from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cpu")

# Disable the NSFW checker filter
pipe.safety_checker = lambda images, clip_input: (images, [False for _ in images])

prompts = [
    "Close-up portrait of a young man with intense teal eyes, pale yellow-green knitted beanie, dark messy hair, contemplative expression, soft diffused lighting, strong focus on eyes, slightly blurred background, realistic digital art",
    "90-degree rotated portrait of a young woman, bright blue eyes, thick eyelashes, beige knit beanie, tousled dark hair, serious contemplative expression, parted lips, soft diffused lighting, muted blue background, realistic digital art",
    "Black and white close-up portrait of an elderly woman with deep wrinkles and thoughtful expression, highly textured skin, dark circles, thin lips, pulled-back hair, dark knitted shawl, soft flattened lighting, introspective mood",
    "Surreal portrait of a young man lying down, dark short hair, thoughtful with closed eyes, textured dark jacket, intense reflection of a fluffy tan and white dog face above, soft diffused lighting, reflective background",
    "Candid photo of a middle-aged man in distress, wearing a dark fur-lined work jacket with hood, standing in corral with calm sheep, pale overcast sky, low angle, soft diffused lighting, mood of loss and grief",
    "High-contrast black and white tight portrait of an elderly man, face full of deep wrinkles and rough textures, dark slightly closed eyes, large nose, wiry hair, dramatic lighting on one side, aged and pensive mood",
    "Intensely detailed black and white close-up of a very old proboscis monkey's face, deeply wrinkled skin, huge bulbous nose, tangled dark hair, prominent ear, flat lighting, highly textured and emotional mood",
    "Collage portrait of diverse people representing various professions, ages, and ethnicities, each in specific character outfits, including model, chef, executive, railway worker, police officer, teacher, fisherman, construction worker, arranged in montage, bright studio lighting, professional photography"
]

for i, prompt in enumerate(prompts, 1):
    print(f"Generating image {i} for prompt:\n{prompt}\n")
    image = pipe(prompt, num_inference_steps=20, guidance_scale=8.0, height=256, width=256).images[0]
    filename = f"generated_image_{i}.png"
    image.save(filename)
    print(f"Saved image {i} as {filename}")