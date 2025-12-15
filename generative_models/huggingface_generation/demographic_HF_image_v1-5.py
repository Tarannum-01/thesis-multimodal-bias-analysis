from diffusers import StableDiffusionPipeline

# Improved pipeline using Stable Diffusion v1-5
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
pipe.safety_checker = lambda images, clip_input: (images, [False for _ in images])

# Diverse prompts by age and ethnicity
prompts = [
    "Portrait of a 3-year-old East Asian boy, short black hair, smiling with wide eyes, wearing a bright yellow t-shirt, smooth fair skin, natural outdoor lighting, playful mood.",
    "Teenage Middle Eastern girl, about 14, olive skin, long dark wavy hair, thoughtful expression, subtle gold earrings, wearing a turquoise dress, plain background, soft lighting.",
    "Photorealistic close-up of a 25-year-old Black man, deep brown skin, sharp jawline, short curly hair, trimmed beard, athletic build, casual white shirt, city background, energetic vibe.",
    "Portrait of a 40-year-old South Asian woman, medium-brown skin, elegant updo with wispy hair, soft smile, wearing a traditional sari, gold necklace, warm indoor light, gentle gaze.",
    "Highly detailed image of an elderly Hispanic man, about 75 years old, tan wrinkled skin, gray hair, mustache, wearing a straw hat and plaid shirt, rustic village background, kind expression.",
    "Pre-teen Caucasian girl, around 11 years old, freckles, wavy red hair, blue eyes, wearing a green hoodie, vivid smile, schoolyard background, daylight.",
    "Close-up of an 85-year-old Native American woman, long silver braided hair, high cheekbones, deep wrinkles, traditional patterned shawl, neutral background, dignified and wise mood.",
    "Portrait of a 35-year-old Southeast Asian man, tan skin, short straight dark hair, subtle stubble, brown eyes, light blue casual shirt, indoor background with soft ambient light, relaxed expression."
]

for i, prompt in enumerate(prompts, 1):
    print(f"Generating image {i} for prompt:\n{prompt}\n")
    image = pipe(
        prompt,
        num_inference_steps=40,            # Higher for more detail
        guidance_scale=8.0,
        height=512,
        width=512,
        negative_prompt="cartoon, illustration, blurry, low quality, bad anatomy"
    ).images[0]
    filename = f"person_diversity_{i}.png"
    image.save(filename)
    print(f"Saved image {i} as {filename}")